from itertools import chain
import torch
from comfy.ldm.modules.attention import optimized_attention


# this code adapted from cubiq's ipadapter implementation
def patch_model_block(model, key, **kwargs):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "attn1" not in to["patches_replace"]:
        to["patches_replace"]["attn1"] = {}
    else:
        to["patches_replace"]["attn1"] = to["patches_replace"]["attn1"].copy()

    if key not in to["patches_replace"]["attn1"]:
        to["patches_replace"]["attn1"][key] = Attn1Replace(**kwargs)
        model.model_options["transformer_options"] = to


class Attn1Replace:
    def __init__(self, **kwargs):
        self.args = kwargs

    def __call__(self, q, k, v, extra_options):
        dtype = q.dtype
        # check if we should disable on certain timesteps
        sigma = (
            extra_options["sigmas"].detach().cpu()[0].item()
            if "sigmas" in extra_options
            else 999999999.9
        )
        should_activate = (
            self.args.get("sigma_end", 0.0)
            <= sigma
            <= self.args.get("sigma_start", 999999999.9)
        )
        # mixingweight, basically
        strength = self.args.get("strength", 1.0)
        out: torch.Tensor
        # if the strength is 1.0, we can optimize slightly by not computing the original attention
        if (strength == 1.0) and should_activate:
            out = 0.0
        else:
            # original attention call before we perturb the queries/values
            out = optimized_attention(q, k, v, extra_options["n_heads"])
        if should_activate:
            # List of [0, 1], [0], [1], ...
            # 0 means conditional, 1 means unconditional
            _cond_or_uncond = extra_options["cond_or_uncond"]
            batches = (q.shape[0] - 1) // len(_cond_or_uncond)
            # essentially "repeat_interleave" on _cond_or_uncond
            cond_or_uncond = [0] + list(
                chain(*([i] * batches for i in _cond_or_uncond))
            )
            # we added the reference image on the 0th index
            q_ref = q[0]
            k_ref = k[0]
            v_ref = v[0]
            # if we're doing NVQG, then the unconditional images should use q_ref instead
            if self.args.get("swap_uncond", False):
                q = torch.stack(
                    [[q_i, q_ref][c] for q_i, c in zip(q, cond_or_uncond)], dim=0
                )
            if self.args.get("swap_cond", False):
                # swap the keys and values in the conditional generation
                k = torch.stack(
                    [[k_ref, k_i][c] for k_i, c in zip(k, cond_or_uncond)], dim=0
                )
                v = torch.stack(
                    [[v_ref, v_i][c] for v_i, c in zip(v, cond_or_uncond)], dim=0
                )
            # call attention
            out_style = optimized_attention(q, k, v, extra_options["n_heads"])
            # mix based on strength
            out = out * (1 - strength) + out_style * strength
        return out.to(dtype=dtype)

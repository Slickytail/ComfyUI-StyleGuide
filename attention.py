import torch
from torch import Tensor
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
    """
    A self-attention processor that swaps keys and values for cond elements, and queries for uncond elements.
    """

    def __init__(self, **kwargs):
        self.args = kwargs
        self.cache_k: Tensor | None = None
        self.cache_v: Tensor | None = None

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, extra_options):
        dtype = q.dtype
        cond_or_uncond: list[int]
        cond_or_uncond = extra_options["cond_or_uncond"]
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
        swap_cond = self.args.get("swap_cond")
        swap_uncond = self.args.get("swap_uncond")
        # this is the pure hydrate pass
        if extra_options.get("hydrate_only"):
            # in case we have multiple images here, let's flatten the batch dimension along the seq dimension
            if swap_cond:
                bl = k.shape[0] * k.shape[1]
                self.cache_k = k.reshape(1, bl, -1)
                self.cache_v = v.reshape(1, bl, -1)
            return optimized_attention(q, k, v, extra_options["n_heads"])
        # original attention call before we perturb the queries/values
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        # okay, actually now there are some extra cases we have to consider
        if should_activate:
            n_ref = extra_options.get("n_ref", 0)
            q_ref = q[:n_ref]
            k_ref = None
            v_ref = None

            if self.cache_k is not None and self.cache_v is not None:
                k_ref = self.cache_k
                v_ref = self.cache_v
            elif n_ref:
                rl = n_ref * k.shape[1]
                k_ref = k[:n_ref].reshape(1, rl, -1)
                v_ref = v[:n_ref].reshape(1, rl, -1)
            elif swap_cond:
                raise ValueError("No cache or reference image provided!")

            batches = (q.shape[0] - n_ref) // len(cond_or_uncond)
            # do cond and uncond in separate calls
            # this is slightly slower since we have to launch 2 kernels instead of one
            # but it's more flexible, since we're not restrained by having the same KV shape for each
            out_cond = None
            if 0 in cond_or_uncond:
                start = n_ref + cond_or_uncond.index(0) * batches
                if swap_cond:
                    q_cond = q[start : start + batches]
                    # actually do the KV injection
                    k_cond = k_ref.expand(batches, -1, -1)
                    v_cond = v_ref.expand(batches, -1, -1)
                    out_cond = optimized_attention(
                        q_cond, k_cond, v_cond, extra_options["n_heads"]
                    )
                else:
                    out_cond = out[start : start + batches]

            # compute attention for uncond batches
            out_uncond = None
            if 1 in cond_or_uncond:
                start = n_ref + cond_or_uncond.index(1) * batches
                if swap_uncond:
                    k = k[start : start + batches]
                    v = v[start : start + batches]
                    out_uncond = 0
                    for i in range(n_ref):
                        q = q_ref[i : i + 1].expand(batches, -1, -1)
                        attn = optimized_attention(q, k, v, extra_options["n_heads"])
                        out_uncond += attn / n_ref
                else:
                    out_uncond = out[start : start + batches]

            out_style = torch.cat([[out_cond, out_uncond][c] for c in cond_or_uncond])
            # mix based on strength
            # mixingweight, basically
            strength = self.args.get("strength", 1.0)
            out[n_ref:] = out[n_ref:] * (1 - strength) + out_style * strength
        return out.to(dtype=dtype)

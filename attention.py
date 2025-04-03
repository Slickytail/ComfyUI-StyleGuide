from itertools import chain
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
        # this is the pure hydrate pass
        if extra_options.get("hydrate_only"):
            # in case we have multiple images here, let's flatten the batch dimension along the seq dimension
            if self.args.get("swap_cond"):
                bl = k.shape[0] * k.shape[1]
                self.cache_k = k.reshape(1, bl, -1)
                self.cache_v = v.reshape(1, bl, -1)
            return optimized_attention(q, k, v, extra_options["n_heads"])
        # mixingweight, basically
        strength = self.args.get("strength", 1.0)
        out: torch.Tensor
        # if the strength is 1.0, we can optimize slightly by not computing the original attention
        if (strength == 1.0) and should_activate:
            out = 0.0  # type: ignore
        else:
            # original attention call before we perturb the queries/values
            out = optimized_attention(q, k, v, extra_options["n_heads"])
        # okay, actually now there are some extra cases we have to consider
        if should_activate:
            n_ref = extra_options.get("n_ref", 0)
            q_ref = q[:n_ref]
            if self.cache_k is not None and self.cache_v is not None:
                k_ref = self.cache_k
                v_ref = self.cache_v
            elif n_ref:
                rl = n_ref * k.shape[1]
                k_ref = k[:n_ref].reshape(1, rl, -1)
                v_ref = v[:n_ref].reshape(1, rl, -1)
            else:
                raise ValueError("No cache or reference image provided!")

            batches = (q.shape[0] - n_ref) // len(cond_or_uncond)
            # we're going to do the cond and uncond attention separately
            # this is slightly inefficient in some cases (ie, if the cached reference is the same shape as the merged one)
            # but in most cases, the cond attention has a different sequence length
            # compute attention for the cond batches
            out_cond = None
            if 0 in cond_or_uncond:
                start = n_ref + cond_or_uncond.index(0) * batches
                q_cond = q[start : start + batches]
                if self.args.get("swap_cond"):
                    # actually do the KV injection
                    k_cond = k_ref.expand(batches, -1, -1)
                    v_cond = v_ref.expand(batches, -1, -1)
                else:
                    # leave the kvs alone
                    k_cond = k[start : start + batches]
                    v_cond = v[start : start + batches]
                out_cond = optimized_attention(
                    q_cond, k_cond, v_cond, extra_options["n_heads"]
                )

            # we'll treat ref as uncond
            # since it should have the same seq size as uncond
            # compute attention for uncond batches
            out_ref = []
            out_uncond = None
            uncond_idx = list(range(n_ref))
            if 1 in cond_or_uncond:
                start = n_ref + cond_or_uncond.index(1) * batches
                uncond_idx = uncond_idx + list(range(start, start + batches))
            if uncond_idx:
                if self.args.get("swap_uncond"):
                    q = q_ref.expand(len(uncond_idx), -1, -1)
                else:
                    q = q[uncond_idx]
                k = k[uncond_idx]
                v = v[uncond_idx]
                out_ref_uncond = optimized_attention(q, k, v, extra_options["n_heads"])
                out_ref, out_uncond = torch.split(
                    out_ref_uncond, [n_ref, out_ref_uncond.shape[0] - n_ref]
                )

            out_style = torch.cat(
                [out_ref] + [[out_cond, out_uncond][c] for c in cond_or_uncond]
            )
            # mix based on strength
            out = out * (1 - strength) + out_style * strength
        return out.to(dtype=dtype)

from dataclasses import dataclass
import torch
from torch import Tensor
from comfy.ldm.modules.attention import optimized_attention


@dataclass
class KVCache:
    k: Tensor
    v: Tensor
    shape: list[int]


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
        self.cache: KVCache | None = None

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
            if swap_cond:
                self.cache = KVCache(
                    k.detach().clone(),
                    v.detach().clone(),
                    shape=extra_options["activations_shape"],
                )
            return optimized_attention(q, k, v, extra_options["n_heads"])
        # original attention call before we perturb the queries/values
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        # okay, actually now there are some extra cases we have to consider
        if should_activate:
            activations_shape = extra_options["activations_shape"]
            n_ref = extra_options.get("n_ref", 0)
            n_ref_total = n_ref
            q_ref = q[:n_ref]
            k_ref = None
            v_ref = None
            s_ref = None
            if self.cache is not None:
                k_ref = self.cache.k
                v_ref = self.cache.v
                s_ref = self.cache.shape
                n_ref_total = k_ref.shape[0]
            elif n_ref:
                k_ref = k[:n_ref]
                v_ref = v[:n_ref]
                s_ref = [n_ref, *activations_shape[1:]]
            elif swap_cond:
                raise ValueError("No cache or reference image provided!")
            attn_mask = None
            get_attn_mask = self.args.get("get_style_mask")
            if get_attn_mask is not None and s_ref is not None:
                if n_ref_total > 1:
                    raise ValueError(
                        "Cannot use multiple references with style-image (KV) masking!"
                    )
                # try to get correct activations size
                attn_mask = get_attn_mask(*s_ref[2:])
                _b, _c, _, _ = attn_mask.shape
                attn_mask = attn_mask.reshape(_b, _c, 1, -1).to(q.device)

            # check for the mask
            get_mask = self.args.get("get_generate_mask")
            if get_mask is not None:
                # try to get correct activations size
                mask = get_mask(*activations_shape[-2:])
                _b, _c, _, _ = mask.shape
                mask = mask.reshape(_b, _c, -1, 1).to(q.device)
                # if we're using tagged pair masks, then we want to keep all channels
                unstyle_mask = 1.0 - mask[
                    :, : (n_ref_total if attn_mask is None else 10000)
                ].sum(dim=1)
            else:
                mask = None
                unstyle_mask = None

            batches = (q.shape[0] - n_ref) // len(cond_or_uncond)
            # do cond and uncond in separate calls
            # this is slightly slower since we have to launch 2 kernels instead of one
            # but it's more flexible, since we're not restrained by having the same KV shape for each
            out_cond = None
            if 0 in cond_or_uncond:
                start = n_ref + cond_or_uncond.index(0) * batches
                if swap_cond:
                    assert k_ref is not None
                    assert v_ref is not None
                    q_cond = q[start : start + batches]
                    out_cond = 0.0
                    # when using cached references, n_ref might be different from the number of kv references we have
                    for i in range(n_ref_total):
                        # actually do the KV injection
                        k_cond = k_ref[i : i + 1].expand(batches, -1, -1)
                        v_cond = v_ref[i : i + 1].expand(batches, -1, -1)
                        _mask = None
                        weight = 1 / n_ref_total
                        # create the mask, or the single-image weight
                        if mask is not None:
                            assert unstyle_mask is not None
                            # tagged pair style mask
                            if attn_mask is not None:
                                # [b, c, q, 1] * [b, c, 1, k] = [b, q, k]
                                _mask = torch.sum(mask * attn_mask, dim=1)
                                _mask = _mask.log().to(dtype=q.dtype, device=q.device)
                                # since KV
                                weight = 1.0 - unstyle_mask
                            # multiple image style mask
                            else:
                                weight = mask[:, i]
                        attn = optimized_attention(
                            q_cond, k_cond, v_cond, extra_options["n_heads"], mask=_mask
                        )
                        out_cond += attn * weight
                    # single mask
                    if unstyle_mask is not None:
                        out_cond += out[start : start + batches] * unstyle_mask
                else:
                    out_cond = out[start : start + batches]

            # compute attention for uncond batches
            out_uncond = None
            if 1 in cond_or_uncond:
                start = n_ref + cond_or_uncond.index(1) * batches
                if swap_uncond:
                    k_uc = k[start : start + batches]
                    v_uc = v[start : start + batches]
                    out_uncond = 0.0
                    n_q = q_ref.shape[0]
                    for i in range(n_q):
                        q_uc = q_ref[i : i + 1].expand(batches, -1, -1)
                        attn = optimized_attention(
                            q_uc, k_uc, v_uc, extra_options["n_heads"]
                        )
                        weight = mask[:, i] if mask is not None else 1 / n_q
                        out_uncond += attn * weight
                    if unstyle_mask is not None:
                        out_uncond += out[start : start + batches] * unstyle_mask
                else:
                    out_uncond = out[start : start + batches]

            out_style = torch.cat([[out_cond, out_uncond][c] for c in cond_or_uncond])
            # mix based on strength
            # mixingweight, basically
            strength = self.args.get("strength", 1.0)
            out[n_ref:] = out[n_ref:] * (1 - strength) + out_style * strength
        return out.to(dtype=dtype)

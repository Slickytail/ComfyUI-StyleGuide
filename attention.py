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
            n_ref = extra_options.get("n_ref", 0)
            q_ref = q[:n_ref]
            k_ref = None
            v_ref = None
            shape_ref = None
            activations_shape = extra_options["activations_shape"]
            if self.cache is not None:
                k_ref = self.cache.k
                v_ref = self.cache.v
                shape_ref = self.cache.shape
            elif n_ref:
                k_ref = k[:n_ref]
                v_ref = v[:n_ref]
                # shape_ref is the same, but with a different batch size
                shape_ref = [n_ref, *activations_shape[1:]]
            elif swap_cond:
                raise ValueError("No cache or reference image provided!")
            # check for the mask
            get_mask = self.args.get("get_mask")

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
                    # for trying the local masking, this is a little test
                    assert get_mask is None
                    assert n_ref == 1
                    assert shape_ref is not None
                    # create the spatial mask
                    # this is basically like inverse ROPE
                    mask = inverse_pe_mask(
                        activations_shape[-2:],
                        shape_ref[-2:],
                        device=q.device,
                        sigma=self.args.get("unmask_sigma", 1),
                        scale=self.args.get("unmask_scale", 1),
                    ).to(dtype=q.dtype)
                    mask = mask.unsqueeze(0).unsqueeze(1)
                    # mask shape is (1, 1, nq, nk)
                    q_cond = q[start : start + batches]
                    out_cond = 0.0
                    # todo: as an optimization, should we check the mask and only pass in the KVs that have nonzero weight
                    for i in range(n_ref):
                        # actually do the KV injection
                        k_cond = k_ref[i : i + 1].expand(batches, -1, -1)
                        v_cond = v_ref[i : i + 1].expand(batches, -1, -1)
                        attn = optimized_attention(
                            q_cond, k_cond, v_cond, extra_options["n_heads"], mask=mask
                        )
                        out_cond += attn
                else:
                    out_cond = out[start : start + batches]

            # compute attention for uncond batches
            out_uncond = None
            if 1 in cond_or_uncond:
                start = n_ref + cond_or_uncond.index(1) * batches
                if swap_uncond:
                    mask = inverse_pe_mask(
                        activations_shape[-2:],
                        activations_shape[-2:],
                        device=q.device,
                        sigma=self.args.get("unmask_sigma", 1),
                        scale=self.args.get("unmask_scale", 1),
                    ).to(dtype=q.dtype)
                    mask = mask.unsqueeze(0).unsqueeze(1)
                    k_uc = k[start : start + batches]
                    v_uc = v[start : start + batches]
                    out_uncond = 0
                    for i in range(n_ref):
                        q_uc = q_ref[i : i + 1].expand(batches, -1, -1)
                        attn = optimized_attention(
                            q_uc, k_uc, v_uc, extra_options["n_heads"], mask=mask
                        )
                        out_uncond += attn
                else:
                    out_uncond = out[start : start + batches]

            out_style = torch.cat([[out_cond, out_uncond][c] for c in cond_or_uncond])
            # mix based on strength
            # mixingweight, basically
            strength = self.args.get("strength", 1.0)
            out[n_ref:] = out[n_ref:] * (1 - strength) + out_style * strength
        return out.to(dtype=dtype)


def inverse_pe_mask(q_shape, k_shape, device, sigma=1, scale=1):
    # okay, wait, the aspect ratio should probably be taken into account
    # anyway, for this first version, we don't care
    d = device
    dt = torch.float16
    pq = torch.zeros((*q_shape, 2), device=d, dtype=dt)
    pq[..., 0] = torch.linspace(0, 1, q_shape[0], device=d, dtype=dt).unsqueeze(1)
    pq[..., 1] = torch.linspace(0, 1, q_shape[1], device=d, dtype=dt).unsqueeze(0)
    pq = pq.reshape(-1, 2)

    pk = torch.zeros((*k_shape, 2), device=d, dtype=dt)
    pk[..., 0] = torch.linspace(0, 1, k_shape[0], device=d, dtype=dt).unsqueeze(1)
    pk[..., 1] = torch.linspace(0, 1, k_shape[1], device=d, dtype=dt).unsqueeze(0)
    pk = pk.reshape(-1, 2)

    # squared distance
    dist = torch.sum((pq.unsqueeze(1) - pk.unsqueeze(0)) ** 2, dim=-1)
    # similarity between two tokens
    inner_product = torch.exp(-dist / sigma**2)  # gaussian
    # inner_product = -torch.sqrt(dist)              # linear
    # inner_product = (dist < sigma**2).to(dtype=dt) # hard
    # the delta-logp should be reduced locally
    # hence the minus
    return -inner_product * scale

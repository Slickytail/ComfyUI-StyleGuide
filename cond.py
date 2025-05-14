import torch


def create_hydrate_cond(c, cond: list, noise, device, model) -> dict:
    cond_tokens = cond[0][0]
    cond_dict = cond[0][1]

    width = noise.shape[3] * 8
    height = noise.shape[2] * 8
    bs = noise.shape[0]

    output = {
        "c_crossattn": cond_tokens.to(device).expand(bs, -1, -1),
        "y": model.encode_adm(
            pooled_output=cond_dict["pooled_output"],
            width=width,
            height=height,
            target_width=width,
            target_height=height,
        )
        .to(device)
        .expand(bs, -1),
        "transformer_options": {
            # so we have to copy most of the stuff here
            # but is there anything that we *dont* want to copy?
            **c["transformer_options"],
            # our attention processor will check for hydrate_only
            "hydrate_only": True,
            "cond_or_uncond": [0],
        },
    }
    if "c_concat" in c:
        output["c_concat"] = torch.zeros_like(noise)
    return output


def create_merged_cond(c: dict, cond: list, n=1, cn_zero_uncond=True) -> dict:
    """
    `c` is the dict containing the actual cond tensors that will be passed to the model.
    `cond` is the output from the conditioning node.

    This function only supports a limited set of cond types.
    We're essentially reimplementing part of the encoding/embedding chain that takes us from a conditioning object to a cond tensor dict.
    And we're constrained by the fact that whatever is in the cond list has to be runnable in the same UNet call as the original one.
    """

    cond_tokens = cond[0][0]
    cond_dict = cond[0][1]
    c = c.copy()
    c["transformer_options"] = c["transformer_options"].copy()
    # our attention implementation will check for n_ref to find out how many ref images there are
    c["transformer_options"]["n_ref"] = n
    # within comfy official code, cond_or_uncond is only used for the SelfAttentionGuidance node
    # combine the actual text encoding tokens
    c["c_crossattn"] = torch.cat(
        (cond_tokens.to(c["c_crossattn"].device).repeat(n, 1, 1), c["c_crossattn"]),
        dim=0,
    )
    # combine the pooled putput
    y = cond_dict["pooled_output"].to(c["y"].device)
    # hacky, but the last part of the vector is height/width/crop values, which are gonna be the same anyway
    y = torch.cat((y, c["y"][0:1, 1280:]), dim=1).repeat(n, 1)
    c["y"] = torch.cat((y, c["y"]), dim=0)
    # combine the controlnet stuff
    # we'll zero out the activations on the new component
    # we're also going to zero out the controlnet on the uncond inputs
    # this increases the strength of the controlnet. otherwise it gets drowned out by the style injection.
    if c.get("control", None) is not None:
        # the batch indices to zero out
        cond_or_uncond = c["transformer_options"]["cond_or_uncond"]
        has_uncond = 1 in cond_or_uncond
        uncond_idx = cond_or_uncond.index(1) if has_uncond else 0
        new = {}
        # the control data is a mapping from [in/middle/out] to lists of tensors: activations to be added inside the model
        old: dict[str, list[torch.Tensor]]
        old = c["control"]
        for k, v in old.items():
            new[k] = []
            for t in v:
                if cn_zero_uncond and has_uncond:
                    true_bs = t.shape[0] // len(cond_or_uncond)
                    t[uncond_idx : uncond_idx + true_bs] = 0.0
                new_activations = torch.zeros_like(t[0]).unsqueeze(0)
                new_activations = torch.cat([new_activations] * n, dim=0)
                out = torch.cat((new_activations, t), dim=0)
                new[k].append(out)
        c["control"] = new

    # this is for handling CosXL
    if c.get("c_concat", None) is not None:
        concat = c["c_concat"]
        extra_concat = torch.zeros(
            (n, *concat.shape[1:]), dtype=concat.dtype, device=concat.device
        )
        c["c_concat"] = torch.cat((extra_concat, concat), dim=0)
    return c

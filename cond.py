import torch


def create_hydrate_cond(c, cond: list, noise, device, model) -> dict:
    cond_tokens = cond[0][0]
    cond_dict = cond[0][1]

    width = noise.shape[3] * 8
    height = noise.shape[2] * 8
    return {
        "c_crossattn": cond_tokens.to(device),
        "y": model.encode_adm(
            pooled_output=cond_dict["pooled_output"],
            width=width,
            height=height,
            target_width=width,
            target_height=height,
        ).to(device),
        "transformer_options": {
            # so we have to copy most of the stuff here
            # but is there anything that we *dont* want to copy?
            **c["transformer_options"],
            "hydrate_only": True,
            "cond_or_uncond": [0],
        },
    }


def create_merged_cond(c: dict, cond: list, cn_zero_uncond=True) -> dict:
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
    c["transformer_options"]["ref_index"] = 0
    # within comfy official code, cond_or_uncond is only used for the SelfAttentionGuidance node
    # combine the actual text encoding tokens
    c["c_crossattn"] = torch.cat(
        (cond_tokens.to(c["c_crossattn"].device), c["c_crossattn"]),
        dim=0,
    )
    # combine the pooled putput
    y = cond_dict["pooled_output"].to(c["y"].device)
    # hacky, but the last part of the vector is height/width/crop values, which are gonna be the same anyway
    y = torch.cat((y, c["y"][0:1, 1280:]), dim=1)
    c["y"] = torch.cat((y, c["y"]), dim=0)
    # combine the controlnet stuff
    # we'll zero out the activations on the new component
    # we're also going to zero out the controlnet on the uncond inputs
    # this increases the strength of the controlnet. otherwise it gets drowned out by the style injection.
    if c.get("control", None) is not None:
        # the batch indices to zero out
        # we increment these because we're going to add the reference on the zero index
        uncond_indices = torch.tensor(
            [
                i + 1
                for i, t in enumerate(c["transformer_options"]["cond_or_uncond"])
                if t
            ]
        )
        new = {}
        # the control data is a mapping from [in/middle/out] to lists of tensors: activations to be added inside the model
        old: dict[str, list[torch.Tensor]]
        old = c["control"]
        for k, v in old.items():
            new[k] = []
            for t in v:
                if cn_zero_uncond:
                    t[uncond_indices] = 0.0
                out = torch.cat((torch.zeros_like(t[0]).unsqueeze(0), t), dim=0)
                # zero out the uncond
                if cn_zero_uncond:
                    out[uncond_indices] = 0.0
                new[k].append(out)
        c["control"] = new
    return c

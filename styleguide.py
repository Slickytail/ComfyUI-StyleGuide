from typing import Callable
import comfy
import torch

from .attention import patch_model_block


def merge_cond(c: dict, cond: list, cn_zero_uncond=True) -> dict:
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
    # treat the reference as uncond
    c["transformer_options"]["cond_or_uncond"] = [1] + c["transformer_options"][
        "cond_or_uncond"
    ]
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
        uncond_indices = torch.tensor(
            [i for i, t in enumerate(c["transformer_options"]["cond_or_uncond"]) if t]
        )
        new = {}
        # the control data is a mapping from [in/middle/out] to lists of tensors: activations to be added inside the model
        old: dict[str, list[torch.Tensor]]
        old = c["control"]
        for k, v in old.items():
            new[k] = []
            for t in v:
                out = torch.cat((torch.zeros_like(t[0]).unsqueeze(0), t), dim=0)
                # zero out the uncond
                if cn_zero_uncond:
                    out[uncond_indices] = 0.0
                new[k].append(out)
        c["control"] = new
    return c


class ApplyVisualStyle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
                "reference_latent": ("LATENT",),
                "reference_cond": ("CONDITIONING",),
                "nvqg_enabled": ("BOOLEAN", {"default": True}),
                "controlnet_use_cfg": ("BOOLEAN", {"default": True}),
                "skip_output_layers": (
                    "INT",
                    {"default": 24, "min": 0, "max": 72, "step": 1},
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The start time % for the stylization.",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The end time % for the stylization.",
                    },
                ),
            }
        }

    CATEGORY = "VisualStylePrompting"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    FUNCTION = "apply_visual_style_prompt"

    def apply_visual_style_prompt(
        self,
        model: comfy.model_patcher.ModelPatcher,
        seed: int,
        reference_latent: dict,
        reference_cond: list,
        nvqg_enabled: bool,
        controlnet_use_cfg: bool,
        skip_output_layers: int = 0,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
    ):

        model = model.clone()
        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(
            start_percent
        )
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(
            end_percent
        )

        # convert the reference latent into the model-internal format
        reference_samples = model.model.latent_format.process_in(
            reference_latent["samples"]
        )

        # todo: support multiple reference images
        # it's fairly clear how to do it for the conditional pass, but not for the uncond (NVQ) pass
        assert reference_samples.shape[0] == 1
        assert len(reference_cond) == 1

        # patch the model's forward function to noise and add the reference samples
        def add_reference(apply_model: Callable, p: dict):
            x = p["input"]
            # t is actually the sigma, not the index
            t = p["timestep"]
            sigma = t.detach().cpu()[0].item()
            # skip the reference injection at some timesteps
            if not (sigma_end <= sigma <= sigma_start):
                return apply_model(x, t, **p["c"])

            # combine the conds
            c = merge_cond(p["c"], reference_cond, cn_zero_uncond=controlnet_use_cfg)
            # noise the reference latent ("stochastic encoding")
            visual = reference_samples.to(x.device)
            noise = comfy.sample.prepare_noise(visual, seed)
            visual = visual + noise.to(x.device) * sigma
            # add it at batch index 0
            # at this point we're past the layer where a batch can be broken up
            # so we can safely add a new element to the batch and know it won't cause problems
            x = torch.concat((visual, x), dim=0)
            # add the timestep
            t = torch.cat((t[0:1], t), dim=0)
            out = apply_model(x, t, **c)
            # remove the useless denoised reference image
            out = out[visual.shape[0] :]
            return out

        # the unet wrapper is the innermost wrapper around the model
        # doing the injection here means that we are less flexible wrt what model/conds are used
        # because we have to convert them to the input format ourselves
        # but it also gives us a lot of control over the performance:
        # comfy won't waste time running the sampler or controlnets on the reference image
        # and we guarantee that it won't be run in a separate batch from the cond/uncond images.
        model.set_model_unet_function_wrapper(add_reference)

        # according to the paper, the NVQ is added in the input and middle blocks
        # we add it to input and middle, and also to the out blocks (up to the skip index)
        if nvqg_enabled:
            # patch input blocks
            for id in [4, 5, 7, 8]:  # id of input_blocks that have self attention
                block_indices = (
                    range(2) if id in [4, 5] else range(10)
                )  # transformer_depth
                for index in block_indices:
                    patch_model_block(
                        model,
                        ("input", id, index),
                        swap_uncond=True,
                        swap_cond=False,
                        sigma_start=sigma_start,
                        sigma_end=sigma_end,
                    )
            # patch middle blocks
            for index in range(10):
                patch_model_block(
                    model,
                    ("middle", 1, index),
                    swap_uncond=True,
                    swap_cond=False,
                    sigma_start=sigma_start,
                    sigma_end=sigma_end,
                )
        # patch output blocks
        blocknum = 0
        for id in range(6):  # there are 6 output blocks
            block_indices = (
                range(2) if id in [3, 4, 5] else range(10)
            )  # the first 3 are depth 10 and the other 3 are 2
            for index in block_indices:
                swap_cond = blocknum >= skip_output_layers
                if blocknum >= skip_output_layers:
                    patch_model_block(
                        model,
                        ("output", id, index),
                        # the blocks that are skipped are included in the NVQG query injection
                        swap_uncond=nvqg_enabled and not swap_cond,
                        swap_cond=swap_cond,
                        sigma_start=sigma_start,
                        sigma_end=sigma_end,
                    )
                blocknum += 1

        return (model,)


NODE_CLASS_MAPPINGS = {
    "ApplyVisualStyle": ApplyVisualStyle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyVisualStyle": "Apply Visual Style Prompting",
}

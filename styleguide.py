import comfy
import torch
from time import sleep

from .attention import patch_model_block


class ApplyVisualStyle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_latent": ("LATENT",),
                "reference_cond": ("CONDITIONING",),
                "nvqg_enabled": ("BOOLEAN", {"default": True}),
                "skip_output_layers": (
                    "INT",
                    {"default": 24, "min": 0, "max": 72, "step": 1},
                ),
            }
        }

    CATEGORY = "VisualStylePrompting"
    RETURN_TYPES = ("MODEL", )
    RETURN_NAMES = ("model", )

    FUNCTION = "apply_visual_style_prompt"

    def apply_visual_style_prompt(
        self,
        model: comfy.model_patcher.ModelPatcher,
        reference_latent,
        reference_cond,
        nvqg_enabled,
        skip_output_layers=0,
    ):

        model = model.clone()
        # convert the reference latent into the model-internal format
        reference_samples = model.model.latent_format.process_in(reference_latent["samples"])

        # todo: support multiple reference images
        # it's fairly clear how to do it for the conditional pass, but not for the uncond (NVQ) pass
        assert reference_samples.shape[0] == 1

        # patch the model's forward function to noise and add the reference samples
        def add_reference(apply_model, p):
            x = p["input"]
            # t is actually the sigma, not the index
            t = p["timestep"]
            c = p["c"].copy()
            c["transformer_options"] = c["transformer_options"].copy()
            # hmm, should we treat the reference generation as cond or uncond?
            # it could have consequences for interaction w/ controlnet & ipadapter
            # for now, treat it as cond
            c["transformer_options"]["cond_or_uncond"] = [0] + c["transformer_options"][
                "cond_or_uncond"
            ]
            # combine the cond
            c["c_crossattn"] = torch.cat((reference_cond[0][0].to(c["c_crossattn"].device), c["c_crossattn"]), dim=0)
            # todo: this actually at this point means we should probably find a different way of injecting the reference image
            # maybe a special CFGGuider?
            y = reference_cond[0][1]["pooled_output"].to(c["y"].device)
            # hacky, but the last part of the vector is height/width/crop values, which are gonna be the same anyway
            y = torch.cat((y, c["y"][0:1, 1280:]), dim=1)
            c["y"] = torch.cat((y, c["y"]), dim=0)
            visual = reference_samples.to(x.device)
            # noise the reference latent ("stochastic encoding")
            visual = visual + torch.randn_like(visual) * t[0]
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

        model.set_model_unet_function_wrapper(add_reference)

        # according to the paper, the NVQ is added in the input and middle blocks
        if nvqg_enabled:
            # patch input blocks
            for id in [4, 5, 7, 8]:  # id of input_blocks that have cross attention
                block_indices = (
                    range(2) if id in [4, 5] else range(10)
                )  # transformer_depth
                for index in block_indices:
                    patch_model_block(
                        model,
                        {"swap_uncond": True, "swap_cond": False},
                        ("input", id, index),
                    )
            # patch middle blocks
            for index in range(10):
                patch_model_block(
                    model,
                    {"swap_uncond": True, "swap_cond": False},
                    ("middle", 1, index),
                )
        # patch output blocks
        blocknum = 0
        for id in range(6):  # there are 6 output blocks
            block_indices = (
                range(2) if id in [3, 4, 5] else range(10)
            )  # the first 3 are depth 10 and the other 3 are 2
            for index in block_indices:
                if blocknum >= skip_output_layers:
                    patch_model_block(
                        model,
                        {"swap_uncond": False, "swap_cond": True},
                        ("output", id, index),
                    )
                blocknum += 1

        return (model, )


NODE_CLASS_MAPPINGS = {
    "ApplyVisualStyle": ApplyVisualStyle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyVisualStyle": "Apply Visual Style Prompting",
}

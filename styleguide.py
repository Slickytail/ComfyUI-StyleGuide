from typing import Callable
import comfy
import torch

from .attention import patch_model_block
from .cond import create_merged_cond, create_hydrate_cond


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
                        "tooltip": "The random seed used for noising the style latent.",
                    },
                ),
                "reference_latent": (
                    "LATENT",
                    {
                        "tooltip": "The encoded latent (without noise) of the style image, at its original size/shape. It should be the same # MP as the generation image, though."
                    },
                ),
                "reference_resized": (
                    "LATENT",
                    {
                        "tooltip": "The encoded latent (without noise) of the style image, resized/cropped to the same shape as the generated image."
                    },
                ),
                "reference_cond": (
                    "CONDITIONING",
                    {
                        "tooltip": "The text conditioning to be passed for the style image's unet call. Doesn't have a big effect."
                    },
                ),
                "nvqg_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether to enable Negative Visual Query Guidance, helping to reduce content leaking.",
                    },
                ),
                "controlnet_use_cfg": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether to zero out controlnet activations on the uncond. "
                        "This increases strength of controlnet, preventing it from "
                        "being drowned out by style transfer.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "How strong the style effect should be.",
                    },
                ),
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
        reference_resized: dict,
        reference_cond: list,
        nvqg_enabled: bool,
        controlnet_use_cfg: bool,
        strength: float = 1.0,
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
        orig_samples = model.get_model_object("latent_format").process_in(
            reference_latent["samples"]
        )
        resized_samples = model.get_model_object("latent_format").process_in(
            reference_resized["samples"]
        )
        # if the original reference is the same shape as the generated image
        # then we can just run it through at the same time,
        # and we don't need to do any KV caching
        needs_separate_hydrate = orig_samples.shape != resized_samples.shape

        # todo: support multiple reference images
        # it's fairly clear how to do it for the conditional pass, but not for the uncond (NVQ) pass
        assert resized_samples.shape[0] == 1
        assert len(reference_cond) == 1

        # patch the model's forward function to noise and add the reference samples
        def add_reference(apply_model: Callable, p: dict):
            x = p["input"]
            # t is actually the sigma, not the index
            t = p["timestep"]
            c = p["c"]
            sigma = t.detach().cpu()[0].item()
            # skip the reference injection at some timesteps
            if not (sigma_end <= sigma <= sigma_start):
                return apply_model(x, t, **p["c"])

            # todo: if nvqg is disabled, we can accept any shape of x by hydrating

            should_hydrate = needs_separate_hydrate and (
                0 in c["transformer_options"]["cond_or_uncond"]
            )
            should_merge = (not needs_separate_hydrate) or (
                1 in c["transformer_options"]["cond_or_uncond"] and nvqg_enabled
            )

            # if we're going to run a cond batch, we might need to hydrate the KV cache
            if should_hydrate:
                # run the reference image through the model
                # add noise
                visual_orig = orig_samples.to(x.device)
                noise_orig = comfy.sample.prepare_noise(visual_orig, seed)
                visual_orig = visual_orig + noise_orig.to(x.device) * sigma
                c_cache = create_hydrate_cond(
                    c, reference_cond, visual_orig, x.device, model.model
                )
                # run the model and discard the output
                apply_model(visual_orig, t[0:1], **c_cache)

            # if we have an uncond, or if we're doing the reference in the same batch, add the resized reference to the batch
            if should_merge:
                # combine the conds
                c = create_merged_cond(
                    c, reference_cond, cn_zero_uncond=controlnet_use_cfg
                )
                # noise the reference latent ("stochastic encoding")
                visual = resized_samples.to(x.device)
                noise = comfy.sample.prepare_noise(visual, seed)
                visual = visual + noise.to(x.device) * sigma
                # add it at batch index 0
                # at this point we're past the layer where a batch can be broken up
                # so we can safely add a new element to the batch and know it won't cause problems
                x = torch.concat((visual, x), dim=0)
                # add the timestep
                t = torch.cat((t[0:1], t), dim=0)

            out = apply_model(x, t, **c)
            # remove the useless denoised reference image, if we added it
            if should_merge:
                out = out[visual.shape[0] :]
            return out

        # the unet wrapper is the innermost wrapper around the model
        # doing the injection here means that we are less flexible wrt what model/conds are used
        # because we have to convert them to the input format ourselves
        # but it also gives us a lot of control over the performance:
        # comfy won't waste time running the sampler or controlnets on the reference image
        # and we guarantee that it won't be run in a separate batch from the cond/uncond images.
        model.set_model_unet_function_wrapper(add_reference)

        style_kwargs = dict(
            sigma_start=sigma_start, sigma_end=sigma_end, strength=strength
        )
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
                        **style_kwargs
                    )
            # patch middle blocks
            for index in range(10):
                patch_model_block(
                    model,
                    ("middle", 0, index),
                    swap_uncond=True,
                    swap_cond=False,
                    **style_kwargs
                )
        # patch output blocks
        blocknum = 0
        for id in range(6):  # there are 6 output blocks
            block_indices = (
                range(2) if id in [3, 4, 5] else range(10)
            )  # the first 3 are depth 10 and the other 3 are 2
            for index in block_indices:
                swap_cond = blocknum >= skip_output_layers
                patch_model_block(
                    model,
                    ("output", id, index),
                    # the blocks that are skipped are included in the NVQG query injection
                    swap_uncond=nvqg_enabled and not swap_cond,
                    swap_cond=swap_cond,
                    **style_kwargs
                )
                blocknum += 1

        return (model,)


NODE_CLASS_MAPPINGS = {
    "ApplyVisualStyle": ApplyVisualStyle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyVisualStyle": "Apply Visual Style Prompting",
}

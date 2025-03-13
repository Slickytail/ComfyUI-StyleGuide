import comfy
import torch

from .utils.attention_functions import VisualStyleProcessor
from .utils.cond_functions import cat_cond


class ApplyVisualStyle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "reference_latent": ("LATENT",),
                "reference_cond": ("CONDITIONING",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "enabled": ("BOOLEAN", {"default": True}),
                "denoise": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 1e-2}),
                "input_blocks": ("BOOLEAN", {"default": False}),
                "skip_input_layers": ("INT", {"default": 24, "min": 0, "max": 48, "step": 1}),
                "middle_block": ("BOOLEAN", {"default": False}),
                "skip_middle_layers": ("INT", {"default": 1, "min": 0, "max": 2, "step": 1}),
                "output_blocks": ("BOOLEAN", {"default": True}),
                "skip_output_layers": ("INT", {"default": 24, "min": 0, "max": 72, "step": 1}),
                "active_blocks": ("STRING", {"default": "0,1,2,3,4,5"})
            },
            "optional": {
                "init_image": ("IMAGE",),
            }
        }

    CATEGORY = "VisualStylePrompting/apply"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "latents")

    FUNCTION = "apply_visual_style_prompt"

    def get_block_choices(self, input_blocks, middle_block, output_blocks):
        block_choices_map = (
            [input_blocks, "input"],
            [middle_block, "middle"],
            [output_blocks, "output"]
        )

        block_choices = []

        for block_choice in block_choices_map:
            if block_choice[0]:
                block_choices.append(block_choice[1])

        return block_choices

    def activate_block_choice(self, key, block_choices):
        return any([block in key for block in block_choices])

    def apply_visual_style_prompt(
        self,
        model: comfy.model_patcher.ModelPatcher,
        clip,
        reference_latent,
        reference_cond,
        positive,
        negative,
        enabled,
        denoise,
        input_blocks,
        middle_block,
        output_blocks,
        active_blocks,
        init_image = None,
        skip_input_layers=0,
        skip_middle_layers=0,
        skip_output_layers=0

    ):

    
        reference_samples = reference_latent["samples"]
        block_choices = self.get_block_choices(input_blocks, middle_block, output_blocks)


        layer_indexes = {
            "input": 0,
            "middle": 0,
            "output": 0
        }

        n_skip_per_block = {
            "input": skip_input_layers,
            "middle": skip_middle_layers,
            "output": skip_output_layers
        }

        block_name_map = {
            "input": "IN",
            "middle": "MID",
            "output": "OUT"
        }

        block_num = 0
        for n, m in model.model.diffusion_model.named_modules():
            if m.__class__.__name__ == "CrossAttention":
                is_self_attn = "attn1" in n  # Only swap self-attention layers
                is_output_block = "output_blocks" in n

                # explicitly only activate clearly output_blocks from skip_output_layers onwards
                is_enabled = is_self_attn and is_output_block and (block_num >= skip_output_layers)
                if is_output_block: 
                    block_num += 1
                    
                if is_enabled:
                    print(n)
                    processor = VisualStyleProcessor(m, enabled=True)
                    m.forward = processor


        positive_cat = cat_cond(clip, reference_cond, positive)
        negative_cat = cat_cond(clip, negative, negative)

        latents = torch.zeros_like(reference_samples) #oh this is because the first one has to be the reference 
        latents = torch.cat([latents] * 2)
        latents[0] = reference_samples

        denoise_mask = torch.ones_like(latents)[:, :1, ...] * denoise

        denoise_mask[0] = 0.0

        return (model, positive_cat, negative_cat, {"samples": latents, "noise_mask": denoise_mask})

NODE_CLASS_MAPPINGS = {
    "ApplyVisualStyle": ApplyVisualStyle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyVisualStyle": "Apply Visual Style Prompting",
}
import comfy
import torch

from .utils.attention_functions import VisualStyleProcessor
from .utils.cond_functions import cat_cond
from .utils.style_functions import color_calibrate


class StochasticEncodeReferenceSDXL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clean_latent": ("LATENT",),   # Clean latent (x₀)
                "sigmas": ("SIGMAS",),         # Sigmas from your sampler
                "timestep": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "max_denoise": ("BOOLEAN", {"default": False}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "VisualStylePrompting/latent"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("noised_reference_latent",)
    FUNCTION = "apply_stochastic_encode_sdxl"

    def apply_stochastic_encode_sdxl(self, clean_latent, sigmas, timestep, max_denoise, enabled):
        if not enabled:
            return clean_latent
        
        x0 = clean_latent["samples"]
        device = x0.device
        dtype = x0.dtype
        
        # Ensure timestep is within valid range:
        if timestep >= len(sigmas):
            timestep = len(sigmas) - 1
        
        sigma = sigmas[timestep].to(device=device, dtype=dtype)
        sigma = sigma.view(sigma.shape[:1] + (1,) * (x0.ndim - 1))
        
        noise = torch.randn_like(x0)
        
        # Apply noise scaling as in SDXL:
        if max_denoise:
            scaled_noise = noise * torch.sqrt(1.0 + sigma ** 2.0)
        else:
            scaled_noise = noise * sigma
        
        # Add the noise to the clean latent:
        x_t = x0 + scaled_noise
        
        return ({"samples": x_t},)

class ColorCalibration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "final_latent": ("LATENT",),
                "reference_latent": ("LATENT",),
                "enabled": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1., "min": 0., "max": 1.0, "step": 0.01}),
            }
        }
    
    CATEGORY = "VisualStylePrompting/color"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("calibrated_latent",)
    FUNCTION = "apply_color_calibration"
    
    def apply_color_calibration(self, final_latent, reference_latent, enabled, intensity):
        if not enabled:
            return final_latent
        samples = final_latent["samples"]
        ref_samples = reference_latent["samples"]
        calibrated = color_calibrate(samples, ref_samples, intensity=intensity)
        return ({"samples": calibrated},)

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
                "adain_q": ("BOOLEAN", {"default": True}),
                "adain_k": ("BOOLEAN", {"default": True}),
                "adain_v": ("BOOLEAN", {"default": False}),
                "nvqg_enabled": ("BOOLEAN", {"default": True}),
                "skip_output_layers": ("INT", {"default": 24, "min": 0, "max": 72, "step": 1}),
                "style_intensity": ("FLOAT", {"default": 1., "min": 0., "max": 1.5, "step": 0.01}),
            },
            "optional": {
                "init_image": ("IMAGE",),
            }
        }

    CATEGORY = "VisualStylePrompting/apply"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "latents")

    FUNCTION = "apply_visual_style_prompt"

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
        adain_q,
        adain_k,
        adain_v,
        nvqg_enabled,
        skip_output_layers=0,
        style_intensity=1.0
    ):

    
        reference_samples = reference_latent["samples"]

        allowed_blocks = ["output_blocks.1", "output_blocks.2", "output_blocks.3", "output_blocks.4", "output_blocks.5"]

        block_num = 0
        for n, m in model.model.diffusion_model.named_modules():
            if m.__class__.__name__ == "CrossAttention":
                is_self_attn = "attn1" in n  # Only swap self-attention layers
                is_output_block = "output_blocks" in n
                
                is_allowed = any(n.startswith(prefix) for prefix in allowed_blocks)

                # explicitly only activate clearly output_blocks from skip_output_layers onwards
                is_enabled = is_self_attn and is_output_block and is_allowed and (block_num >= skip_output_layers)
                if is_output_block: 
                    block_num += 1

                if is_enabled:
                    print(n)
                    processor = VisualStyleProcessor(m, enabled=True, adain_queries=adain_q, adain_keys=adain_k, adain_values=adain_v, nvqg_enabled=nvqg_enabled, style_intensity=style_intensity)
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
    "StochasticEncodeReferenceSDXL": StochasticEncodeReferenceSDXL,
    "ColorCalibration": ColorCalibration,
    "ApplyVisualStyle": ApplyVisualStyle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StochasticEncodeReferenceSDXL": "Stochastic Encode Reference SDXL",
    "ColorCalibration": "Color Calibration",
    "ApplyVisualStyle": "Apply Visual Style Prompting",
}
import torch
from torch import Tensor
from comfy.samplers import KSAMPLER
from comfy.k_diffusion.sampling import to_d
from tqdm.auto import trange


def adain_latent(x: Tensor, mean: Tensor, std: Tensor, interp: float = 1.0) -> Tensor:
    """
    Adjust a latent to have the target statistics (mean/std).
    Set interp<1.0 to only partially adjust.
    """
    mean = mean.to(x.device)
    std = std.to(x.device)
    mean_input = x.mean(dim=(2, 3), keepdim=True)
    std_input = x.std(dim=(2, 3), keepdim=True)

    xp = (x - mean_input) / std_input * std + mean
    return xp * interp + x * (1 - interp)


class ColorGradeLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("LATENT", {"tooltip": "The latent to be color graded"}),
                "reference": ("LATENT", {"tooltip": "The reference latent"}),
            },
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "grade"

    CATEGORY = "VisualStylePrompting"
    DESCRIPTION = "Take a latent and color grade it to match the reference latent"

    def grade(self, input, reference):
        mean_reference = reference["samples"].mean(dim=(0, 2, 3), keepdim=True)
        std_reference = (
            reference["samples"].std(dim=(2, 3), keepdim=True).mean(dim=0, keepdim=True)
        )

        output_latent = adain_latent(input["samples"], mean_reference, std_reference)

        return ({"samples": output_latent},)


class ColorGradeEulerSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "reference": ("LATENT", {"tooltip": "The reference latent"}),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The start time % for the colorgrading.",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The end time % for the colorgrading.",
                    },
                ),
                "rate": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "How much to adjust the colors at each step (compared to full normalization). This can help to reduce artifacts,"
                        "by giving the model time to correct the errors caused by adjusting the latent. In general, only about 0.4 is needed to to match colors.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("SAMPLER",)

    FUNCTION = "create_sampler"
    CATEGORY = "VisualStylePrompting"
    DESCRIPTION = "sampler to color grade the latent to match the reference latent"

    def create_sampler(
        self,
        model,
        reference,
        start_percent: float,
        end_percent: float,
        rate: float,
    ):
        # inside the model, the latent has a different scale/mean than what the VAE processes
        # so in order to get the right mean/variance, we need to rescale the reference latent
        ref = model.model.latent_format.process_in(reference["samples"])
        mean_reference = ref.mean(dim=(0, 2, 3), keepdim=True)
        # we actually want the mean standard deviation per image -- if we try to take the std over multiple images, it'll be much higher than expected.
        std_reference = ref.std(dim=(2, 3), keepdim=True).mean(dim=0, keepdim=True)

        if end_percent <= start_percent:
            raise ValueError("End must be greater than start.")

        # custom euler sampler to add the calibration step
        @torch.no_grad()
        def sample_euler_calibrate(
            model,
            x,
            sigmas,
            extra_args=None,
            callback=None,
            disable=None,
            s_churn=0.0,
            s_tmin=0.0,
            s_tmax=float("inf"),
            s_noise=1.0,
        ):
            """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
            i_start = int(start_percent * len(sigmas))
            i_end = int(end_percent * len(sigmas))
            extra_args = {} if extra_args is None else extra_args
            s_in = x.new_ones([x.shape[0]])
            for i in trange(len(sigmas) - 1, disable=disable):
                if s_churn > 0:
                    gamma = (
                        min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
                        if s_tmin <= sigmas[i] <= s_tmax
                        else 0.0
                    )
                    sigma_hat = sigmas[i] * (gamma + 1)
                else:
                    gamma = 0
                    sigma_hat = sigmas[i]

                if gamma > 0:
                    eps = torch.randn_like(x) * s_noise
                    x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
                denoised = model(x, sigma_hat * s_in, **extra_args)
                d = to_d(x, sigma_hat, denoised)
                # colorgrade the latent
                if i_start <= i <= i_end:
                    denoised = adain_latent(
                        denoised, mean_reference, std_reference, interp=rate
                    )

                if callback is not None:
                    callback(
                        {
                            "x": x,
                            "i": i,
                            "sigma": sigmas[i],
                            "sigma_hat": sigma_hat,
                            "denoised": denoised,
                        }
                    )

                # Euler method
                x = denoised + d * sigmas[i + 1]
            return x

        # instantiate a KSampler with this custom function
        sampler = KSAMPLER(sample_euler_calibrate)
        return (sampler,)


NODE_CLASS_MAPPINGS = {
    "ColorGradeEulerSampler": ColorGradeEulerSampler,
    "ColorGradeLatent": ColorGradeLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorGradeEulerSampler": "Sampler Euler Colorgrade",
    "ColorGradeLatent": "Colorgrade Latent",
}

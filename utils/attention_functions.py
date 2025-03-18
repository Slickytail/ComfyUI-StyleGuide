from comfy.ldm.modules.attention import default, optimized_attention, optimized_attention_masked

from .style_functions import adain, swapping_attention

class VisualStyleProcessor(object):
    def __init__(self, 
        module_self, 
        style_intensity: float = 1.0,
        enabled: bool = True, 
        adain_queries: bool = True,
        adain_keys: bool = True,
        adain_values: bool = False,
        nvqg_enabled: bool = True,      # new: enable NVQG
        nvqg_weight: float = 0.5         # new: strength of negative guidance
    ):
        self.module_self = module_self
        self.style_intensity = style_intensity
        self.enabled = enabled
        self.adain_queries = adain_queries
        self.adain_keys = adain_keys
        self.adain_values = adain_values
        self.nvqg_enabled = nvqg_enabled
        self.nvqg_weight = nvqg_weight

    def __call__(self, x, context, value, mask=None):
        return self.visual_style_forward(x, context, value, mask)

    def visual_style_forward(self, x, context, value, mask=None):
        # Compute the query from input x
        q = self.module_self.to_q(x)
        context = default(context, x)
        k = self.module_self.to_k(context)
        v = self.module_self.to_v(context)

        if self.enabled:
            if self.adain_queries:
                q = adain(q)
            if self.adain_keys:
                k = adain(k)
            if self.adain_values:
                v = adain(v)

            # Perform swapping on k,v using reference style
            k, v = swapping_attention(k, v, style_intensity=self.style_intensity)
        
        # Compute positive attention using the modified q,k,v
        if mask is None:
            positive_attn = optimized_attention(q, k, v, self.module_self.heads)
        else:
            positive_attn = optimized_attention_masked(q, k, v, self.module_self.heads, mask)

        # If NVQG is enabled, compute a negative branch and combine.
        if self.nvqg_enabled:
            # Compute a "negative" query.
            # Here we simply recompute the raw query without applying adain (or with a different transformation).
            q_negative = self.module_self.to_q(x)
            # Optionally, you could skip any style normalization on q_negative.
            if mask is None:
                negative_attn = optimized_attention(q_negative, k, v, self.module_self.heads)
            else:
                negative_attn = optimized_attention_masked(q_negative, k, v, self.module_self.heads, mask)
            
            # Combine the two attention outputs.
            # This follows a CFG-style formulation: final = (1 + w)*positive - w*negative.
            combined_attn = (1 + self.nvqg_weight) * positive_attn - self.nvqg_weight * negative_attn
        else:
            combined_attn = positive_attn

        return self.module_self.to_out(combined_attn)

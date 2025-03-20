from comfy.ldm.modules.attention import default, optimized_attention, optimized_attention_masked

from .style_functions import adain, swapping_attention, get_attention

class VisualStyleProcessor(object):
    def __init__(self, 
        module_self, 
        style_intensity: float = 1.0,
        enabled: bool = True, 
        adain_queries: bool = True,
        adain_keys: bool = True,
        adain_values: bool = False,
        nvqg_enabled: bool = True,      # new: enable NVQG
        nvqg_weight: float = 1.0         # new: strength of negative guidance
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
        context = default(context, x)
        q = self.module_self.to_q(context)
        k = self.module_self.to_k(context)
        v = self.module_self.to_v(context)
        k_orig = k
        v_orig = v
        if self.enabled:
            if self.adain_queries:
                q = adain(q)
            if self.adain_keys:
                k = adain(k)
            if self.adain_values:
                v = adain(v)

            # Perform swapping on k,v using reference style
            k, v = swapping_attention(k, v, style_intensity=self.style_intensity)
        
        q_style = get_attention(q,0)
        k_style = get_attention(k,0)
        v_style = get_attention(v,0)

        q_text = get_attention(q,1)
        k_text = get_attention(k,1)
        v_text = get_attention(v,1)

        # Compute positive attention using the modified q,k,v

        positive_attn = optimized_attention(q, k_style, v_style, self.module_self.heads)

        # If NVQG is enabled, compute a negative branch and combine.
        if self.nvqg_enabled:
            print("enabled")
            # Compute a "negative" query.
            # Here we simply recompute the raw query without applying adain (or with a different transformation).
            q_negative = self.module_self.to_q(x)
            # Optionally, you could skip any style normalization on q_negative.
            if mask is None:
                negative_attn = optimized_attention(q_negative, k_orig, v_orig, self.module_self.heads)
            else:
                negative_attn = optimized_attention_masked(q_negative, k, v, self.module_self.heads, mask)       

            # Combine the two attention outputs.
            # This follows a CFG-style formulation: final = (1 + w)*positive - w*negative.
            combined_attn = (1 + self.nvqg_weight) * positive_attn - self.nvqg_weight * negative_attn
        else:
            combined_attn = positive_attn

        return self.module_self.to_out(combined_attn)

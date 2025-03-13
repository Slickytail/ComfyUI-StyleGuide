from comfy.ldm.modules.attention import default, optimized_attention, optimized_attention_masked, attention_basic
from .style_functions import adain, concat_first, swapping_attention, q_content


class VisualStyleProcessor(object):
    def __init__(self, 
        module_self, 
        keys_scale: float = 1.0,
        enabled: bool = True, 
        adain_queries: bool = True,
        adain_keys: bool = True,
        adain_values: bool = False 
    ):
        self.module_self = module_self
        self.keys_scale = keys_scale
        self.enabled = enabled
        self.adain_queries = adain_queries
        self.adain_keys = adain_keys
        self.adain_values = adain_values

    def __call__(self, x, context, value, mask=None):
        return self.visual_style_forward(x, context, value, mask)

    def visual_style_forward(self, x, context, value, mask=None):
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

            k, v = swapping_attention(k, v)
        
        # q = q_content(q)

        if mask is None:
            out = optimized_attention(q, k, v, self.module_self.heads)
        else:
            out = optimized_attention_masked(q, k, v, self.module_self.heads, mask)
        return self.module_self.to_out(out)


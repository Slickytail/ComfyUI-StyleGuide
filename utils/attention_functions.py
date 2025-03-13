from comfy.ldm.modules.attention import default, optimized_attention, optimized_attention_masked, attention_basic
from .style_functions import adain, concat_first, swapping_attention


class VisualStyleProcessor(object):
    def __init__(self, 
        module_self, 
        keys_scale: float = 1.0,
        enabled: bool = True, 
        enabled_animatediff: bool = False,
        adain_queries: bool = False,
        adain_keys: bool = False,
        adain_values: bool = False 
    ):
        self.module_self = module_self
        self.keys_scale = keys_scale
        self.enabled = enabled
        self.enabled_animatediff = enabled
        self.adain_queries = adain_queries
        self.adain_keys = adain_keys
        self.adain_values = adain_values

    def __call__(self, x, context, value, mask=None):
        return self.visual_style_forward(x, context, value, mask)

    def visual_style_forward(self, x, context, value, mask=None):
        q = self.module_self.to_q(x)
        context = default(context, x)
        k = self.module_self.to_k(context)
        print(context)
        if value is not None:
            v = self.module_self.to_v(value)
            del value
        else:
            v = self.module_self.to_v(context)

        if self.enabled:
            if self.adain_queries:
                q = adain(q)
            if self.adain_keys:
                k = adain(k)
            if self.adain_values:
                v = adain(v)
            
            k, v = swapping_attention(k, v)

            #k = concat_first(k, -2, self.keys_scale)
            #v = concat_first(v, -2)

        if mask is None:
            out = optimized_attention(q, k, v, self.module_self.heads)
        else:
            out = optimized_attention_masked(q, k, v, self.module_self.heads, mask)
        return self.module_self.to_out(out)


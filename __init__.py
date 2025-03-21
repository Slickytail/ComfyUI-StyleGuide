from . import styleguide, calibration

NODE_CLASS_MAPPINGS = {
    **styleguide.NODE_CLASS_MAPPINGS,
    **calibration.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **styleguide.NODE_DISPLAY_NAME_MAPPINGS,
    **calibration.NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

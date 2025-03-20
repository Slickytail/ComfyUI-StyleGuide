import torch

from einops import rearrange
from dataclasses import dataclass

T = torch.Tensor


def color_calibrate(pred: T, ref: T, intensity: float = 1.0, eps: float = 1e-5) -> T:
    """
    Calibrate the colors of 'pred' to match the channel-wise mean and std of 'ref'.
    Both pred and ref should be tensors of shape (batch, channels, height, width).
    """
    pred_mean = pred.mean(dim=[2, 3], keepdim=True)
    pred_std = pred.std(dim=[2, 3], keepdim=True)
    ref_mean = ref.mean(dim=[2, 3], keepdim=True)
    ref_std = ref.std(dim=[2, 3], keepdim=True)
    
    calibrated = (pred - pred_mean) / (pred_std + eps) * ref_std + ref_mean
    out = (1.0 - intensity) * pred + intensity * calibrated
    return out


def expand_first(feat: T, scale=1., ) -> T:
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat

def q_content(query,chunk_size=2):
    chunk_length = query.size()[0] // chunk_size  # [text-condition, null-condition]
    original_image_index = [1] * chunk_length  # [0 0 0 0 0]
    query = rearrange(query, "(b f) d c -> b f d c", f=chunk_length)
    query = query[:, original_image_index]  # ref to all
    query = rearrange(query, "b f d c -> (b f) d c")
    return query

def get_attention(attn, index, chunk_size=2):
    #index of 0 is style and index of 1 is text
    chunk_length = attn.size()[0] // chunk_size #should be 2 
    image_index = [index] * chunk_length  # this then becomes [0,0]
    attn_reshaped = rearrange(attn, "(b f) d c -> b f d c", f=chunk_length)
    attn_swapped = attn_reshaped[:, image_index]  # select reference style
    attn_swapped = rearrange(attn_swapped, "b f d c -> (b f) d c")
    return attn_swapped

def swapping_attention(query, key, value, style_intensity=1.0, chunk_size=2):
    """
    Interpolate between the original key/value and the swapped key/value.
    When style_intensity is 0, the original key/value are preserved.
    When style_intensity is 1, the swapped key/value are fully used.
    """
    # Compute the swapped keys and values as before.
    chunk_length = key.size()[0] // chunk_size #should be 2 
    reference_image_index = [0] * chunk_length  # this then becomes [0,0]
    
    query_reshaped = rearrange(query, "(b f) d c -> b f d c", f=chunk_length)
    query_swapped = query_reshaped[:, reference_image_index]  # select reference style
    query_swapped = rearrange(query_swapped, "b f d c -> (b f) d c")
    
    # Rearrange and extract swapped key.
    key_reshaped = rearrange(key, "(b f) d c -> b f d c", f=chunk_length)
    key_swapped = key_reshaped[:, reference_image_index]  # select reference style
    key_swapped = rearrange(key_swapped, "b f d c -> (b f) d c")
    
    # Rearrange and extract swapped value.
    value_reshaped = rearrange(value, "(b f) d c -> b f d c", f=chunk_length)
    value_swapped = value_reshaped[:, reference_image_index]
    value_swapped = rearrange(value_swapped, "b f d c -> (b f) d c")
    
    # Perform linear interpolation between the original and swapped features.
    new_key = (1.0 - style_intensity) * key + style_intensity * key_swapped
    new_value = (1.0 - style_intensity) * value + style_intensity * value_swapped
    new_query = (1.0 - style_intensity) * query + style_intensity * query_swapped
    return new_query, new_key, new_value
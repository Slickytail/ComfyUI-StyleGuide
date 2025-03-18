import torch

from einops import rearrange
from dataclasses import dataclass

T = torch.Tensor


def color_calibrate(pred: T, ref: T, eps: float = 1e-5) -> T:
    """
    Calibrate the colors of 'pred' to match the channel-wise mean and std of 'ref'.
    Both pred and ref should be tensors of shape (batch, channels, height, width).
    """
    pred_mean = pred.mean(dim=[2, 3], keepdim=True)
    pred_std = pred.std(dim=[2, 3], keepdim=True)
    ref_mean = ref.mean(dim=[2, 3], keepdim=True)
    ref_std = ref.std(dim=[2, 3], keepdim=True)
    
    calibrated = (pred - pred_mean) / (pred_std + eps) * ref_std + ref_mean
    return calibrated


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

def swapping_attention(key, value, chunk_size=2):
    chunk_length = key.size()[0] // chunk_size  # [text-condition, null-condition]
    reference_image_index = [0] * chunk_length  # [0 0 0 0 0]
    key = rearrange(key, "(b f) d c -> b f d c", f=chunk_length)
    key = key[:, reference_image_index]  # ref to all
    key = rearrange(key, "b f d c -> (b f) d c")
    value = rearrange(value, "(b f) d c -> b f d c", f=chunk_length)
    value = value[:, reference_image_index]  # ref to all
    value = rearrange(value, "b f d c -> (b f) d c")

    return key, value
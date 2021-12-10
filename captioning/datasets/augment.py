import random

import numpy as np
import torch

from .nb_SparseImageWarp import sparse_image_warp

def time_warp(spec: np.ndarray, W: int = 5):
    spec = torch.as_tensor(spec).transpose(0, 1).unsqueeze(0)
    spec_len = spec.shape[2]
    num_freq = spec.shape[1]
    device = spec.device

    y = num_freq // 2
    horizontal_line_at_ctr = spec[0][y]
    assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
    assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device),
                         torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3).squeeze(0).transpose(0, 1).numpy()


def freq_mask(spec: np.ndarray, F: int = 30, num_masks: int = 1, replace_with_zero: bool = False):
    """spec: [T, F]"""
    cloned = spec.copy()
    num_channels = cloned.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        if (replace_with_zero):
            cloned[:, f_zero:mask_end] = 0
        else:
            cloned[:, f_zero:mask_end] = cloned.mean()

    return cloned


def time_mask(spec: np.ndarray, T: int = 40, num_masks: int = 1, replace_with_zero: bool = False):
    cloned = spec.copy()
    len_spectro = cloned.shape[0]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero):
            cloned[t_zero:mask_end, :] = 0
        else:
            cloned[t_zero:mask_end, :] = cloned.mean()
    return cloned


def spec_augment(timemask: bool = True, num_timemask: int = 2, 
        freqmask: bool = True, num_freqmask: int = 2, timewarp: bool = False, 
        F: int = 15, W: int = 40, T: int = 30, p: float = 0.2):
    def wrapper(spec: np.ndarray):
        # spec: [T, F]
        if random.random() < p:
            if timemask and num_timemask > 0:
                spec = time_mask(spec, T=T, num_masks=num_timemask)
            if freqmask and num_freqmask > 0:
                spec = freq_mask(spec, F=F, num_masks=num_freqmask)
            if timewarp:
                spec = time_warp(spec, W=W)
        return spec
    return wrapper


def gaussian_noise(x, snr=30, mean=0):
    E_x = (x**2).sum() / x.shape[0]
    noise = torch.empty_like(x).normal_(mean, std=1)
    E_noise = (noise**2).sum() / noise.shape[0]
    alpha = np.sqrt(E_x / (E_noise * pow(10, snr / 10)))
    x = x + alpha * noise
    return x

def random_crop(spec, size: int = 1000, p: float = 0.2):
    time, freq = spec.shape
    if time <= size or random.random() > p:
        return spec
    hi = time - size
    # start_ind = torch.empty(1, dtype=torch.long).random_(0, hi).item()
    start_ind = np.random.randint(0, hi)
    spec = spec[start_ind:start_ind + size, :]
    return spec

def time_roll(x, mean=0, std=10):
    """
    x: either wave or spectrogram
    """
    # shift = torch.empty(1).normal_(mean, std).int().item()
    shift = int(np.random.normal(mean, std))
    # x = torch.roll(x, shift, dims=0)
    x = np.roll(x, shift, axis=0)
    return x

if __name__ == "__main__":
    random.seed(1)
    x = np.random.randn(501, 64)
    augment_function = spec_augment(timemask=False, timewarp=False)
    x = augment_function(x)
    print(x.shape)

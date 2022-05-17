import random

import numpy as np
import torch

from .nb_SparseImageWarp import sparse_image_warp


def time_warp(spec: np.ndarray, W: int = 5):
    spec = torch.as_tensor(spec).transpose(0, 1).unsqueeze(0)
    num_rows = spec.shape[2]
    spec_len = spec.shape[1]
    device = spec.device

    pt = (num_rows - 2* W) * torch.rand([1], dtype=torch.float) + W # random point along the time axis
    src_ctr_pt_freq = torch.arange(0, spec_len // 2)  # control points on freq-axis
    src_ctr_pt_time = torch.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
    src_ctr_pts = torch.stack((src_ctr_pt_freq, src_ctr_pt_time), dim=-1)
    src_ctr_pts = src_ctr_pts.float().to(device)

    # Destination
    w = 2 * W * torch.rand([1], dtype=torch.float) - W# distance
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = torch.stack((dest_ctr_pt_freq, dest_ctr_pt_time), dim=-1)
    dest_ctr_pts = dest_ctr_pts.float().to(device)

    # warp
    source_control_point_locations = torch.unsqueeze(src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = torch.unsqueeze(dest_ctr_pts, 0)  # (1, v//2, 2)
    warped_spectro, dense_flows = sparse_image_warp(spec, 
                                                    source_control_point_locations, 
                                                    dest_control_point_locations)
    warped_spectro = warped_spectro.squeeze(3).squeeze(0).transpose(0, 1).numpy()
    return warped_spectro


def freq_mask(spec: np.ndarray, F: int = 30, num_masks: int = 1, replace_with_zero: bool = True):
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


def time_mask(spec: np.ndarray, T: int = 40, num_masks: int = 1, replace_with_zero: bool = True):
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


def gaussian_noise(snr=30, mean=0):
    def wrapper(waveform):
        x = waveform
        E_x = (x ** 2).sum() / x.shape[0]
        noise = torch.empty_like(x).normal_(mean, std=1)
        E_noise = (noise ** 2).sum() / noise.shape[0]
        alpha = np.sqrt(E_x / (E_noise * pow(10, snr / 10)))
        x = x + alpha * noise
        return x
    return wrapper


def random_crop(size: int = 1000, p: float = 0.2):
    def wrapper(spec):
        time, freq = spec.shape
        if time <= size or random.random() > p:
            return spec
        hi = time - size
        start_ind = np.random.randint(0, hi)
        spec = spec[start_ind: start_ind + size, :]
        return spec
    return wrapper


def time_roll(mean=0, std=10):
    def wrapper(x):
        """
        x: either wave or spectrogram
        """
        # shift = torch.empty(1).normal_(mean, std).int().item()
        shift = int(np.random.normal(mean, std))
        # x = torch.roll(x, shift, dims=0)
        x = np.roll(x, shift, axis=0)
        return x
    return wrapper


if __name__ == "__main__":
    random.seed(1)
    x = np.random.randn(501, 64)
    augment_function = spec_augment(timemask=False, timewarp=False)
    x = augment_function(x)
    print(x.shape)

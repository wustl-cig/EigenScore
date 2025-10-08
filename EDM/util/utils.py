import numpy as np
import PIL.Image
import torch
import os
import re


def normalize_np(img):
    """Normalize img in arbitrary range to [0, 1]"""
    img -= np.min(img)
    img /= np.max(img)
    return img


def normalize(img):
    """Normalize img in arbitrary range to [0, 1]"""
    img -= img.min()
    img /= img.max()
    return img


def clear_gray(img):

    img = img.detach().cpu().squeeze().numpy()
    return normalize_np(img)


def clear(img):

    img = img.detach().cpu().squeeze().numpy()
    # img = normalize_np(np.transpose(img.squeeze().detach().cpu().numpy(), (1, 2, 0)))
    return normalize_np(np.transpose(img, (1, 2, 0)))


def clear_255(img):
    # detach and make the image ready for visualization and saving range [-1, 1]
    img = (
        np.transpose(normalize_np((img).squeeze().detach().cpu().numpy()), (1, 2, 0))
        * 255
    )
    return img


def scale(width, height, img):

    w, h = img.size
    if width == w and height == h:
        return img
    # img = PIL.Image.fromarray(img)
    ww = width if width is not None else w
    hh = height if height is not None else h
    img = img.resize((ww, hh), PIL.Image.Resampling.LANCZOS)
    return np.array(img)


def denoise_net(
    net, im, time=0, num_steps=100, rho=1, sigma_min=0.002, sigma_max=80, device="cuda"
):

    step_indices = torch.arange(num_steps, dtype=torch.float64)
    t_steps = torch.flip(
        (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        )
        ** rho,
        [0],
    )

    t_hat = t_steps[time]
    im = (im * 2) - 1   
    sigma = t_hat
    noisemap = torch.randn_like(im)
    nim = im + sigma * noisemap
    denoised = net(nim, t_hat.to(device)).to(torch.float64)
    return (denoised + 1) / 2, sigma, (nim + 1) / 2


def parse_str_list(s):
    item_list = s.split(",") if s else []
    return item_list


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def get_net_list(rootpath, net_list):
    list_model = []
    for net in net_list:
        list_model.append(os.path.join(rootpath, net) + ".pkl")
    return list_model


def shufflerow(tensor, axis):
    row_perm = (
        torch.rand(tensor.shape[: axis + 1]).argsort(axis).to(tensor.device)
    )  # get permutation indices
    for _ in range(tensor.ndim - axis - 1):
        row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(
        *[1 for _ in range(axis + 1)], *(tensor.shape[axis + 1 :])
    )  # reformat this for the gather operation
    return tensor.gather(axis, row_perm)


def get_mask(batch_size=1, acs_lines=30, total_lines=320, R=1):
    # Overall sampling budget
    num_sampled_lines = total_lines // R

    # Get locations of ACS lines
    # !!! Assumes k-space is even sized and centered, true for fastMRI
    center_line_idx = torch.arange(
        (total_lines - acs_lines) // 2, (total_lines + acs_lines) // 2
    )

    # Find remaining candidates
    outer_line_idx = torch.cat(
        [
            torch.arange(0, (total_lines - acs_lines) // 2),
            torch.arange((total_lines + acs_lines) // 2, total_lines),
        ]
    )
    random_line_idx = shufflerow(
        outer_line_idx.unsqueeze(0).repeat([batch_size, 1]), 1
    )[:, : num_sampled_lines - acs_lines]
    # random_line_idx = outer_line_idx[torch.randperm(outer_line_idx.shape[0])[:num_sampled_lines - acs_lines]]

    # Create a mask and place ones at the right locations
    mask = torch.zeros((batch_size, total_lines))
    mask[:, center_line_idx] = 1.0
    mask[
        torch.arange(batch_size).repeat_interleave(random_line_idx.shape[-1]),
        random_line_idx.reshape(-1),
    ] = 1.0

    return mask


from typing import List, Optional

import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft  # type: ignore


def fft2c_old(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Whether to include normalization. Must be one of ``"backward"``
            or ``"ortho"``. See ``torch.fft.fft`` on PyTorch 1.9.0 for details.
    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    if norm not in ("ortho", "backward"):
        raise ValueError("norm must be 'ortho' or 'backward'.")
    normalized = True if norm == "ortho" else False

    data = ifftshift(data, dim=[-3, -2])
    data = torch.fft(data, 2, normalized=normalized)
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_old(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Whether to include normalization. Must be one of ``"backward"``
            or ``"ortho"``. See ``torch.fft.ifft`` on PyTorch 1.9.0 for
            details.
    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    if norm not in ("ortho", "backward"):
        raise ValueError("norm must be 'ortho' or 'backward'.")
    normalized = True if norm == "ortho" else False

    data = ifftshift(data, dim=[-3, -2])
    data = torch.ifft(data, 2, normalized=normalized)
    data = fftshift(data, dim=[-3, -2])

    return data


def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.
    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.
    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


# Helper functions


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.
    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.
    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.
    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.
    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for s, d in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.
    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.
    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


def fft2_m(x):
    """FFT for multi-coil"""
    return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
    """IFFT for multi-coil"""
    return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))


class SinglecoilMRI_comp:
    def __init__(self, image_size, mask):
        self.image_size = image_size
        self.mask = mask

    def A(self, x):
        return fft2_m(x) * self.mask

    def A_dagger(self, x):
        return ifft2_m(x)

    def A_T(self, x):
        return self.A_dagger(x)

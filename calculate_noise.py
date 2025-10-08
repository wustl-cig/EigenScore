import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms as T


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    """
    if schedule_name == "linear":
        # Linear schedule from 0.0001 to 0.02
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def calculate_noise_level(timestep, total_steps=1000):
    """
    Calculate the noise level at a specific timestep.
    """
    if timestep >= total_steps:
        raise ValueError(
            f"Timestep {timestep} is out of range. Must be less than {total_steps}"
        )

    # Get the beta schedule
    betas = get_named_beta_schedule("linear", total_steps)

    # Calculate alphas
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    # Calculate noise level
    sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

    # Get noise level at specific timestep
    noise_level = sqrt_recipm1_alphas_cumprod[timestep]

    return noise_level


if __name__ == "__main__":
    # Show noise levels for various timesteps
    timesteps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
    print("Noise levels at different timesteps (1000 steps):")
    print("-----------------------------------")

    for t in timesteps:
        noise_level = calculate_noise_level(t)
        print(f"Timestep {t:3d}: {noise_level:.6f}")







import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, NamedTuple, Union
import time


def _power_iteration_batched(
    model: torch.nn.Module,
    nim_batch: torch.Tensor,
    eigvecs: torch.Tensor,
    mask_batch: torch.Tensor,
    n_ev: int,
    double_precision: bool,
    sigma: float,
):
    if torch.cuda.is_available():
        torch.cuda.synchronize(nim_batch.device)

    with torch.no_grad():
        # === CENTRAL DIFFERENCE COMPUTATION ===
        # Create inputs for forward (+eigvecs) and backward (-eigvecs) perturbations
        # This implements central difference: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
        inputs = torch.cat([nim_batch + eigvecs, nim_batch - eigvecs], dim=0)

        # Convert to double precision if requested for higher numerical accuracy
        if double_precision:
            inputs = inputs.double()
        batch_flat = inputs.reshape(-1, *inputs.shape[2:])
        outputs = model(batch_flat, sigma)

        # Reshape back to separate +/- perturbations
        outputs = outputs.reshape_as(inputs)

        # Split into positive and negative perturbation results
        out_plus, out_minus = outputs.chunk(2, dim=0)

        # Apply mask to restrict computation to relevant image regions
        out_plus = out_plus * mask_batch
        out_minus = out_minus * mask_batch

        # Compute central difference: (f(x+h) - f(x-h)) / 2
        # This approximates the Jacobian-vector product (Jv)
        Ab = 0.5 * (out_plus - out_minus)

        # === VECTOR NORMALIZATION ===
        # Compute L2 norms of each eigenvector candidate for normalization
        # Reshape to [B, n_ev, spatial_dims] and compute norm over spatial dimensions
        norms = (Ab * mask_batch).reshape(Ab.shape[0], Ab.shape[1], -1).norm(dim=2)

        # Normalize eigenvectors to unit length to prevent exponential growth/decay
        # Add small epsilon (1e-12) to prevent division by zero
        eigvecs_next = Ab / (
            norms.clamp_min(1e-12).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )

        # Reapply mask to ensure eigenvectors remain zero outside relevant regions
        eigvecs_next = eigvecs_next * mask_batch

        # === QR ORTHOGONALIZATION ===
        # Extract dimensions for matrix operations
        B = eigvecs_next.shape[0]  # Batch size
        P = int(  # Total number of spatial elements per eigenvector
            torch.prod(torch.tensor(eigvecs_next.shape[2:], device=eigvecs_next.device))
        )

        # Reshape eigenvectors into matrix form: [B, P, n_ev]
        # Each column represents one eigenvector as a flattened vector
        M = eigvecs_next.reshape(B, n_ev, P).transpose(1, 2)

        # Perform QR decomposition to orthogonalize eigenvectors
        # Q contains orthonormal vectors, R contains the transformation matrix
        Q, R = torch.linalg.qr(M, mode="reduced")

        # === DETERMINANT SIGN CORRECTION ===
        # Compute determinant of R matrix to check orientation
        det = torch.prod(torch.linalg.diagonal(R, dim1=-2, dim2=-1), dim=-1)

        # If determinant is negative, flip sign to maintain consistent orientation
        neg = det < 0
        if neg.any():
            Q[neg] *= -1

        # Final normalization to ensure unit vectors (numerical stability)
        Q = Q / Q.norm(dim=-2, keepdim=True).clamp_min(1e-12)

        # Reshape back to original eigenvector format [B, n_ev, C, H, W]
        eigvecs_ortho = Q.transpose(1, 2).reshape_as(eigvecs_next)

    if torch.cuda.is_available():
        torch.cuda.synchronize(nim_batch.device)

    return eigvecs_ortho, norms


def get_eigvecs_batched(
    model: torch.nn.Module,
    nim_batch: torch.Tensor,
    mask: torch.Tensor,
    n_ev: int,
    sigma: float,
    device: torch.device,
    c: float = 1,
    iters: int = 50,
    double_precision: bool = False,
    verbose: bool = False,
):
    nim_batch = nim_batch.to(device)
    if double_precision:
        nim_batch = nim_batch.double()

    # Expand the mask to the batch size
    if mask.dim() == 3:
        mask_batch = mask.unsqueeze(0).expand(nim_batch.shape[0], -1, -1, -1)
    else:
        mask_batch = mask
    mask_batch = mask_batch.to(device)

    B = nim_batch.shape[0]

    # Repeat the noisy input for each eigenvector
    nim_repeated = nim_batch.unsqueeze(1).expand(-1, n_ev, -1, -1, -1)

    # Initialize and normalize random vectors for power iteration
    eigvecs = torch.randn_like(nim_repeated) * mask_batch.unsqueeze(1) * c
    prev = eigvecs.clone()
    prev = prev / prev.reshape(B, n_ev, -1).norm(dim=2).clamp_min(1e-12).unsqueeze(
        -1
    ).unsqueeze(-1).unsqueeze(-1)

    times_reached = torch.zeros(B, dtype=torch.int64, device=device)
    converged_mask = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(iters):
        eigvecs, jacprod_norm = _power_iteration_batched(
            model,
            nim_repeated,
            eigvecs,
            mask_batch.unsqueeze(1).expand(-1, n_ev, -1, -1, -1),
            n_ev,
            double_precision,
            sigma,
        )

        scaled_vals = jacprod_norm / c * (sigma**2)
        sorted_vals, indices = torch.sort(
            scaled_vals, dim=1, descending=True, stable=True
        )
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, n_ev)
        eigvecs = eigvecs[batch_indices, indices]

        evs_now = eigvecs.reshape(B, n_ev, -1).to(torch.float64)
        evs_prev = prev.reshape(B, n_ev, -1).to(torch.float64)
        corr_mat = torch.einsum("bik,bjk->bij", evs_now, evs_prev)
        diag_sum = torch.diagonal(corr_mat, dim1=-2, dim2=-1).sum(dim=1)
        times_reached = torch.where(
            diag_sum > 2.94, times_reached + 1, torch.zeros_like(times_reached)
        )
        converged_mask = converged_mask | (times_reached >= 5)
        if converged_mask.all():
            prev = eigvecs.clone()
            eigvecs = eigvecs * c
            break

        prev = eigvecs.clone()
        eigvecs = eigvecs * c

    eigvals = jacprod_norm / c * (sigma**2)
    eigvecs = eigvecs / c

    # Final per-sample sort of eigenvalues/eigenvectors
    sorted_vals, indices = torch.sort(eigvals, dim=1, descending=True, stable=True)
    B, K = indices.shape
    batch_indices = torch.arange(B, device=eigvals.device).unsqueeze(1).expand(-1, K)
    eigvecs = eigvecs[batch_indices, indices]
    eigvals = sorted_vals

    if not converged_mask.all():
        evs_now = eigvecs.reshape(B, n_ev, -1).to(torch.float64)
        evs_prev = prev.reshape(B, n_ev, -1).to(torch.float64)
        corr_mat = torch.einsum("bik,bjk->bij", evs_now, evs_prev)
        diag_sum = torch.diagonal(corr_mat, dim1=-2, dim2=-1).sum(dim=1)
        converged_mask = converged_mask | (diag_sum >= 2.94)

    b_masked_mmse = torch.tensor(0, device=device)
    corr_history = None
    mmse_placeholder = torch.tensor(0, device=device)
    return (
        eigvecs,
        eigvals,
        b_masked_mmse,
        sigma,
        corr_history,
        mmse_placeholder,
        converged_mask,
    )

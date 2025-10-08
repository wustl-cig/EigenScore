from PIL import Image
import click
from tqdm import tqdm
import pickle
from torchvision import transforms as T
import os
import torch
import gc
import pandas as pd
from datetime import datetime

from EDM.util.utils import *
from EDM import dnnlib

toTensor = T.ToTensor()
from eigenscore_calculate import *
from calculate_noise import calculate_noise_level

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP (NumPy / SciPy / skimage)
os.environ["MKL_NUM_THREADS"] = "1"  # Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def toim(img: torch.Tensor) -> Image.Image:
    if not (img.min() >= 0 and img.max() <= 1):
        img = ((img + 1) / 2).clip(0, 1)
    untensor = T.ToPILImage()
    return untensor(img)


@click.command()
@click.option(
    "--outpath",
    help="Where to save the outputs",
    default="/Test",
    type=str,
)
@click.option(
    "--dataset_path",
    help="path to testset",
    default="Z_Z_FINAL/Data/celeba-32",
    type=str,
)
@click.option(
    "--model_path",
    help="path to model",
    default="/Models",
    type=str,
)
@click.option("--gpu-ids", help="which GPU to use", default=0, type=int)
@click.option("--img_size", help="size of the test images", default=64, type=int)
@click.option("--n_ev", help="number of eigenvectors", default=3, type=int)
@click.option("--const", help="constant", default=1, type=float)
@click.option("--iters", help="number of iterations", default=50, type=int)
@click.option("--double_precision", help="double precision", default=True, type=bool)
@click.option("--dataset_name", help="dataset name", default="ffhq", type=str)
@click.option("--save_img", help="save the denoised image", default=False, type=int)
@click.option(
    "--verbose",
    help="print the denoising metric of each image",
    default=False,
    type=bool,
)
@click.option("--num_samples", help="number of samples", default=1000, type=int)
@click.option("--model_name", help="model name", default="celeba.pkl", type=str)
@click.option("--id_name", help="id name", default="celeba", type=str)
@click.option(
    "--n_repetitions", help="number of repetitions for each image", default=20, type=int
)
@click.option(
    "--rep_batch",
    help="batch size for repetitions per power-iteration call",
    default=20,
    type=int,
)
def main(**sampler_kwargs):
    opts = dnnlib.EasyDict(sampler_kwargs)

    # Validate paths
    if not os.path.exists(opts.dataset_path):
        raise ValueError(f"Dataset path does not exist: {opts.dataset_path}")
    if not os.path.exists(opts.model_path):
        raise ValueError(f"Model path does not exist: {opts.model_path}")

    # Create output directories if they don't exist
    os.makedirs(opts.outpath, exist_ok=True)

    time_steps = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    # time_steps = [800]
    device_str = f"cuda:{opts.gpu_ids}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model_path = os.path.join(opts.model_path, opts.model_name)
    print(f"Loading model from {model_path}...")
    print(f"output path: {opts.outpath}")
    if not os.path.exists(model_path):
        raise ValueError(f"Model file does not exist: {model_path}")

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)["ema"].to(device)
        model.eval()
        print("model loaded")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

    # load num_samples of images and scale them
    imgs_path = os.listdir(opts.dataset_path)
    imgs_path = imgs_path[: opts.num_samples]

    # Initialize DataFrame to store results
    results_data = []
    convergence_data = []  # Store convergence information
    start_time = datetime.now()
    n_repetitions = (
        opts.n_repetitions
    )  # Number of converged calculations needed per image
    # Note: total_operations is now variable since we stop when we get n_repetitions converged results
    expected_operations = (
        len(time_steps) * len(imgs_path) * n_repetitions
    )  # Minimum operations if all converge
    current_operation = 0

    for time_step in time_steps:
        for index, path in enumerate(tqdm(imgs_path)):

            img = Image.open(os.path.join(opts.dataset_path, path)).convert("RGB")
            img_name = imgs_path[index].split(".")[0]

            print(f"\n{'=' * 80}")
            print(
                f"Processing time step {time_step} ({time_steps.index(time_step) + 1}/{len(time_steps)})"
            )
            print(f"Processing image {index + 1}/{len(imgs_path)}: {imgs_path[index]}")
            print(f"{'=' * 80}\n")

            im = toTensor(img)
            im = (im * 2) - 1
            im = im.to(device)

            # Continue until we get n_repetitions converged calculations
            converged_count = 0
            attempt_count = 0
            max_attempts = (
                n_repetitions * 5
            )  # Safety mechanism to prevent infinite loops

            while converged_count < n_repetitions and attempt_count < max_attempts:
                attempt_count += 1
                current_operation += 1

                # Calculate estimated time remaining (approximate since total ops is now variable)
                elapsed_time = datetime.now() - start_time
                avg_time_per_op = (
                    elapsed_time.total_seconds() / current_operation
                    if current_operation > 0
                    else 0
                )

                remaining = n_repetitions - converged_count
                batch_size = min(int(getattr(opts, "rep_batch", 8)), remaining)
                print(
                    f"Attempt {attempt_count} (Converged: {converged_count}/{n_repetitions}) | batch {batch_size}"
                )
                if avg_time_per_op > 0:
                    print(f"Average time per operation: {avg_time_per_op:.2f} seconds")

                # Build a batch of different noises
                sigma = calculate_noise_level(time_step)
                sigma = torch.tensor(sigma, device=device)
                nim_batch = []
                for _ in range(batch_size):
                    noisemap = torch.randn_like(im)
                    nim_b = im + sigma * noisemap
                    nim_batch.append(nim_b)
                nim_batch = torch.stack(nim_batch, dim=0)

                # Optional one-time save of denoised output
                if opts.save_img and converged_count == 0:
                    with torch.no_grad():
                        sample_out = model(nim_batch[0:1], sigma)
                    sample_img = toim(
                        sample_out[0] if sample_out.dim() == 4 else sample_out
                    )
                    output_path = os.path.join(
                        opts.outpath, f"output_{time_step}_{img_name}.png"
                    )
                    sample_img.save(output_path)

                selected_coords = {
                    "x1": 0,
                    "x2": opts.img_size,
                    "y1": 0,
                    "y2": opts.img_size,
                }

                mask = torch.zeros_like(im, device=device)
                mask[
                    :,
                    selected_coords["y1"] : selected_coords["y2"],
                    selected_coords["x1"] : selected_coords["x2"],
                ] = 1

                eigvecs_b, eigvals_b, _, sigma_ret, _, _, converged_flags = (
                    get_eigvecs_batched(
                        model,
                        nim_batch,
                        mask,
                        opts.n_ev,
                        sigma,
                        device,
                        c=opts.const,
                        iters=opts.iters,
                        double_precision=opts.double_precision,
                        verbose=opts.verbose,
                    )
                )

                # Collect converged samples from the batch
                for b in range(batch_size):
                    if converged_flags[b].item():
                        converged_count += 1
                        # Store results
                        result_dict = {
                            "image_name": img_name,
                            "time_step": time_step,
                            "repetition": converged_count,
                            "v1": eigvals_b[b, 0].detach().cpu().numpy(),
                            "v2": eigvals_b[b, 1].detach().cpu().numpy(),
                            "v3": eigvals_b[b, 2].detach().cpu().numpy(),
                            "converged": True,
                            "attempt_number": attempt_count,
                        }
                        results_data.append(result_dict)
                        # print(
                        #    f"✓ Converged calculation {converged_count}/{n_repetitions} (batch idx {b})"
                        # )
                        if converged_count >= n_repetitions:
                            break
                else:
                    # If no break occurred (i.e., not enough yet), just continue
                    pass

                # Clear GPU memory
                del nim_batch, eigvecs_b, eigvals_b
                torch.cuda.empty_cache()
                gc.collect()

            # Check if we reached the desired number of converged results
            if converged_count < n_repetitions:
                # Only add to convergence_data if we failed to get required converged results
                convergence_dict = {
                    "image_name": img_name,
                    "time_step": time_step,
                    "converged_count": converged_count,
                    "required_count": n_repetitions,
                    "total_attempts": attempt_count,
                    "iterations": opts.iters,
                    "sigma": calculate_noise_level(time_step),
                    "const": opts.const,
                }
                convergence_data.append(convergence_dict)
                print(
                    f"Warning: Only got {converged_count}/{n_repetitions} converged results for {img_name} at timestep {time_step}"
                )
                print(
                    f"Stopped after {attempt_count} attempts to prevent infinite loop"
                )

            img.close()

    # Convert results to DataFrame
    df = pd.DataFrame(results_data)

    # Create a multi-index DataFrame with time steps and repetitions as columns
    # Format: time_step_repetition (e.g., 100_rep1, 100_rep2, etc.)
    df["time_step_rep"] = (
        df["time_step"].astype(str) + "_rep" + df["repetition"].astype(str)
    )

    pivot_df = df.pivot(
        index="image_name", columns="time_step_rep", values=["v1", "v2", "v3"]
    )

    # --- NEW FORMAT: Timesteps as top-level columns, repetitions and v1/v2/v3 as subcolumns ---
    # Prepare mapping from timestep to noise level
    timestep_to_noise = {ts: calculate_noise_level(ts) for ts in time_steps}
    # Create new column names: e.g., 100(0.05)_rep1
    df["timestep_label"] = df["time_step"].apply(
        lambda ts: f"{ts}({timestep_to_noise[ts]:.5f})"
    )
    df["timestep_rep_label"] = (
        df["timestep_label"] + "_rep" + df["repetition"].astype(str)
    )

    # Create a hierarchical structure: timestep -> repetition -> eigenvalues
    df_for_pivot = df.set_index(["image_name", "time_step", "repetition"])[
        ["v1", "v2", "v3"]
    ]
    df_wide = df_for_pivot.unstack(level=[1, 2])

    # Save to Excel (hierarchical format)
    excel_path2 = os.path.join(
        opts.outpath,
        f"{opts.id_name}_{opts.dataset_name}_eigenvalues_by_timestep_reps.xlsx",
    )
    df_wide.to_excel(excel_path2)
    print(f"Results saved to {excel_path2}")

    # --- CONVERGENCE DATA: Save convergence information ---
    if convergence_data:  # Only create Excel file if there are non-convergent cases
        convergence_df = pd.DataFrame(convergence_data)
        excel_path4 = os.path.join(
            opts.outpath,
            f"{opts.id_name}_{opts.dataset_name}_non_convergent_cases.xlsx",
        )
        convergence_df.to_excel(excel_path4, index=False)
        print(f"Non-convergent cases saved to {excel_path4}")
    else:
        print(
            "No convergence issues detected - all eigenvector calculations converged!"
        )
        convergence_df = pd.DataFrame()  # Empty dataframe for summary statistics

    # Print summary statistics
    print("\nResults Summary:")
    for time_step in time_steps:
        time_data = df[df["time_step"] == time_step]
        print(f"\nTime Step {time_step}:")
        print(f"  Average eigenvalues across all repetitions:")
        print(
            f"    v1: {time_data['v1'].mean():.6f} (std: {time_data['v1'].std():.6f})"
        )
        print(
            f"    v2: {time_data['v2'].mean():.6f} (std: {time_data['v2'].std():.6f})"
        )
        print(
            f"    v3: {time_data['v3'].mean():.6f} (std: {time_data['v3'].std():.6f})"
        )
        print(
            f"  Total samples: {len(time_data)} ({len(time_data) // n_repetitions} images × {n_repetitions} repetitions)"
        )

    print(f"\nConfiguration:")
    print(f"  Output path: {opts.outpath}")
    print(f"  Model: {opts.model_name}")
    print(f"  Dataset: {opts.dataset_name}")
    print(f"  ID name: {opts.id_name}")
    print(f"  Images processed: {len(imgs_path)}")
    print(f"  Repetitions per image: {n_repetitions}")
    print(f"  Total eigenvalue calculations: {len(results_data)}")
    print(f"  Iterations per calculation: {opts.iters}")
    print(f"  Time steps: {time_steps}")

    # Overall convergence statistics (saved to Excel only)
    total_non_converged = len(
        convergence_df
    )  # convergence_df only contains non-convergent cases
    total_calculations = len(df)  # df contains all calculations
    total_converged = total_calculations - total_non_converged


if __name__ == "__main__":
    main()

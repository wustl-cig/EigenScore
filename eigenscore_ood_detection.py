import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import argparse
from itertools import product
from util import (
    extract_eigenvalue_sums,
    compute_eigen_aggregated_per_timestep,
    load_and_extract_features,
)


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(
        description="Out-of-Distribution Detection using EigenScore method"
    )

    # Dataset arguments
    parser.add_argument(
        "-id",
        "--id_dataset",
        type=str,
        default="cifar10",
        help="In-distribution dataset name (default: cifar10)",
    )
    parser.add_argument(
        "-ood",
        "--ood_dataset",
        type=str,
        default="cifar100",
        help="Out-of-distribution dataset name (default: cifar100)",
    )

    # File path arguments
    parser.add_argument(
        "-e",
        "--eigenvalues_path",
        type=str,
        default="/Results",
        help="Path to the directory of eigenvalues files ",
    )

    # Timestep arguments
    parser.add_argument(
        "-time",
        "--timesteps",
        nargs="+",
        type=int,
        default=[100, 150, 200, 250, 300],
        help="List of timesteps to use (default: [100, 150, 200, 250, 300])",
    )

    # Model parameters
    parser.add_argument(
        "-a",
        "--aggregation_method",
        type=str,
        default="mean",
        choices=["mean", "median", "all"],
        help="Aggregation method for features (default: mean)",
    )
    parser.add_argument(
        "-rep",
        "--num_repetitions",
        type=int,
        default=20,
        help="Number of repetitions to use (default: 20)",
    )

    # Split parameters
    parser.add_argument(
        "-t",
        "--test_fraction",
        type=float,
        default=0.5,
        help="Fraction of data to use for testing (default: 0.5)",
    )

    return parser.parse_args()


class BasicOODDetector:
    """
    Basic Out-of-Distribution Detection using standardized feature sum for eigenvalue-based features.
    """

    def __init__(
        self,
        id_file_path,
        ood_file_path,
        timesteps_to_use=None,
        aggregation_method="mean",
        num_repetitions=None,
        id_df=None,
        ood_df=None,
    ):
        self.id_file_path = id_file_path
        self.ood_file_path = ood_file_path
        self.timesteps_to_use = timesteps_to_use
        self.aggregation_method = aggregation_method
        self.num_repetitions = num_repetitions
        self.scaler = StandardScaler()
        self._id_df = id_df
        self._ood_df = ood_df

    def _fit_scaler(self, fit_features):
        """Fit StandardScaler on given ID features."""
        features_clean = np.nan_to_num(fit_features, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler.fit(features_clean)
        return self.scaler

    def _transform_features(self, features):
        """Transform features using the fitted StandardScaler."""
        features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(features_clean)

    def run_detection_with_validation(self, test_fraction=0.5):
        """Run OOD detection with train/validation/test splits."""
        # Load features
        id_features, ood_features = load_and_extract_features(
            self.id_file_path,
            self.ood_file_path,
            timesteps_to_use=self.timesteps_to_use,
            aggregation_method=self.aggregation_method,
            num_repetitions=self.num_repetitions,
            id_df=self._id_df,
            ood_df=self._ood_df,
        )

        # Split ID data: 50% for train (also used as validation), 50% for test
        id_train, id_test = train_test_split(
            id_features, test_size=test_fraction, random_state=42
        )
        id_val = id_train

        # Split OOD data: 50% for validation, 50% for test
        ood_val, ood_test = train_test_split(
            ood_features, test_size=test_fraction, random_state=42
        )

        # Fit StandardScaler on training data
        self._fit_scaler(id_train)
        id_train_std = self._transform_features(id_train)
        id_val_std = self._transform_features(id_val)
        id_test_std = self._transform_features(id_test)
        ood_val_std = self._transform_features(ood_val)
        ood_test_std = self._transform_features(ood_test)

        # Compute scores as sum of standardized features
        id_train_scores = id_train_std.sum(axis=1)
        id_val_scores = id_val_std.sum(axis=1)
        id_test_scores = id_test_std.sum(axis=1)
        ood_val_scores = ood_val_std.sum(axis=1)
        ood_test_scores = ood_test_std.sum(axis=1)

        # Validation AUC
        val_scores = np.concatenate([id_val_scores, ood_val_scores])
        val_labels = np.concatenate(
            [np.zeros(len(id_val_scores)), np.ones(len(ood_val_scores))]
        )
        val_auc = roc_auc_score(val_labels, val_scores)

        # Final test evaluation
        test_scores = np.concatenate([id_test_scores, ood_test_scores])
        test_labels = np.concatenate(
            [np.zeros(len(id_test_scores)), np.ones(len(ood_test_scores))]
        )
        test_auc = roc_auc_score(test_labels, test_scores)

        return {
            "test_auc": test_auc,
            "validation_auc": val_auc,
        }


def main_single_aggregation(**kwargs):
    """Run OOD detection with a single aggregation method."""
    # Extract parameters from kwargs or use defaults
    id_dataset = kwargs.get("id_dataset", "cifar10")
    ood_dataset = kwargs.get("ood_dataset", "cifar100")
    custom_timesteps = kwargs.get("custom_timesteps", [100, 150, 200, 250, 300])
    aggregation_method = kwargs.get("aggregation_method", "mean")
    num_repetitions = kwargs.get("num_repetitions", 20)
    test_fraction = kwargs.get("test_fraction", 0.5)
    id_df = kwargs.get("id_df")
    ood_df = kwargs.get("ood_df")

    # Generate file paths if not provided
    if "id_file" in kwargs:
        id_file = kwargs["id_file"]

    if "ood_file" in kwargs:
        ood_file = kwargs["ood_file"]

    # Print initial configuration
    print(f"\nStarting OOD Detection...")
    print(f"Loading data for {id_dataset} (ID) vs {ood_dataset} (OOD)")
    print(f"Using timesteps: {custom_timesteps}")
    print(f"Aggregation method: {aggregation_method}")

    # Run OOD detection
    detector = BasicOODDetector(
        id_file,
        ood_file,
        timesteps_to_use=custom_timesteps,
        aggregation_method=aggregation_method,
        num_repetitions=num_repetitions,
        id_df=id_df,
        ood_df=ood_df,
    )

    results = detector.run_detection_with_validation(test_fraction=test_fraction)

    final_auc = results["test_auc"]
    val_auc = results["validation_auc"]

    # Print results for visibility
    print(f"\n=== OOD Detection Results ===")
    print(f"ID Dataset: {id_dataset}")
    print(f"OOD Dataset: {ood_dataset}")
    print(f"Timesteps: {custom_timesteps}")
    print(f"Aggregation Method: {aggregation_method}")
    print(f"Number of Repetitions: {num_repetitions}")
    print(f"Validation AUROC: {val_auc:.4f}")
    print(f"Test AUROC: {final_auc:.4f}")
    print("=" * 30)

    if kwargs:
        return f"{final_auc:.4f}", f"{val_auc:.4f}"
    return final_auc, val_auc


def run_multiple_with_validation_preload(
    id_file,
    ood_file,
    timesteps_grid,
    aggregation_methods,
    num_repetitions,
    id_dataset,
    ood_dataset,
    output_dir,
    test_fraction=0.5,
):
    """Run parameter optimization with validation-based approach and preloaded DataFrames."""
    # Preload once
    id_df = pd.read_excel(id_file, index_col=0, header=[0, 1, 2])
    ood_df = pd.read_excel(ood_file, index_col=0, header=[0, 1, 2])

    results = []

    for agg_method in aggregation_methods:
        for ts in timesteps_grid:
            auc, val_auc = main_single_aggregation(
                id_file=id_file,
                ood_file=ood_file,
                custom_timesteps=ts,
                aggregation_method=agg_method,
                num_repetitions=num_repetitions,
                id_dataset=id_dataset,
                ood_dataset=ood_dataset,
                id_df=id_df,
                ood_df=ood_df,
                test_fraction=test_fraction,
            )
            results.append(
                {
                    "ID Dataset": id_dataset,
                    "OOD Dataset": ood_dataset,
                    "Timestep(s)": str(ts),
                    "Aggregation": agg_method,
                    "Validation_AUROC": val_auc,
                    "Test_AUROC": auc,
                }
            )

    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{id_dataset}_{ood_dataset}_kl_ood_validation_evaluation.xlsx"
    )
    df.to_excel(output_path, index=False)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Generate default file paths if not provided

    id_file = f"{args.eigenvalues_path}/{args.id_dataset}_{args.id_dataset}_eigenvalues_by_timestep_reps.xlsx"

    ood_file = f"{args.eigenvalues_path}/{args.id_dataset}_{args.ood_dataset}_eigenvalues_by_timestep_reps.xlsx"

    # Call main_single_aggregation with command line arguments
    main_single_aggregation(
        id_dataset=args.id_dataset,
        ood_dataset=args.ood_dataset,
        id_file=id_file,
        ood_file=ood_file,
        custom_timesteps=args.timesteps,
        aggregation_method=args.aggregation_method,
        num_repetitions=args.num_repetitions,
        test_fraction=args.test_fraction,
    )

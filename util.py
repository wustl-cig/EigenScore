import pandas as pd
import numpy as np


def compute_eigen_aggregated_per_timestep(
    df, timesteps, repetitions, aggregation_method="mean"
):
    """
    Build an (n_images, n_timesteps) matrix where each column is the aggregated
    (mean/median across repetitions) eigenvalue sum (v1+v2+v3) for a timestep.
    """
    per_timestep_cols = []
    for timestep in timesteps:
        values_per_rep = []
        for rep in repetitions:
            v1 = df.loc[:, ("v1", timestep, rep)].to_numpy()
            v2 = df.loc[:, ("v2", timestep, rep)].to_numpy()
            v3 = df.loc[:, ("v3", timestep, rep)].to_numpy()
            summed = v1 + v2 + v3
            values_per_rep.append(summed)

        mat = np.vstack(values_per_rep).T  # (n_images, n_reps_present)
        if aggregation_method == "median":
            agg = np.nanmedian(mat, axis=1)
        else:
            agg = np.nanmean(mat, axis=1)
        per_timestep_cols.append(agg)

    return np.stack(per_timestep_cols, axis=1)


def extract_eigenvalue_sums(df, timesteps, repetitions, aggregation_method="mean"):
    """
    Extract sum of eigenvalues (v1+v2+v3) for specified timesteps and repetitions.
    """
    if aggregation_method == "all":
        features_list = []
        for image_name in df.index:
            image_features = []
            for timestep in timesteps:
                for rep in repetitions:
                    eigenvalue_sum = (
                        df.loc[image_name, ("v1", timestep, rep)]
                        + df.loc[image_name, ("v2", timestep, rep)]
                        + df.loc[image_name, ("v3", timestep, rep)]
                    )
                    image_features.append(eigenvalue_sum)
            features_list.append(image_features)
        features = np.array(features_list)
    else:
        features_list = []
        for image_name in df.index:
            row_features = []
            for timestep in timesteps:
                timestep_sums = []
                for rep in repetitions:
                    eigenvalue_sum = (
                        df.loc[image_name, ("v1", timestep, rep)]
                        + df.loc[image_name, ("v2", timestep, rep)]
                        + df.loc[image_name, ("v3", timestep, rep)]
                    )
                    timestep_sums.append(eigenvalue_sum)

                if aggregation_method == "median":
                    aggregated_value = np.median(timestep_sums)
                else:
                    aggregated_value = np.mean(timestep_sums)
                row_features.append(aggregated_value)
            features_list.append(row_features)
        features = np.array(features_list)

    return features


def load_and_extract_features(
    id_file_path,
    ood_file_path,
    timesteps_to_use=None,
    aggregation_method="mean",
    num_repetitions=None,
    id_df=None,
    ood_df=None,
):
    """
    Load Excel files and extract eigenvalue sum features for specified timesteps.
    """
    # Load Excel files with MultiIndex columns unless preloaded DataFrames are provided
    id_df = (
        id_df
        if id_df is not None
        else pd.read_excel(id_file_path, index_col=0, header=[0, 1, 2])
    )
    ood_df = (
        ood_df
        if ood_df is not None
        else pd.read_excel(ood_file_path, index_col=0, header=[0, 1, 2])
    )

    # Extract available timesteps and repetitions
    all_timesteps = sorted(id_df.columns.levels[1])
    all_repetitions = sorted(id_df.columns.levels[2])

    # Determine which repetitions to use
    repetitions_to_use = (
        all_repetitions[:num_repetitions] if num_repetitions else all_repetitions
    )

    # Determine which timesteps to use
    timesteps_to_use_final = timesteps_to_use if timesteps_to_use else all_timesteps

    # Extract features for ID and OOD data
    id_features = extract_eigenvalue_sums(
        id_df,
        timesteps_to_use_final,
        repetitions_to_use,
        aggregation_method=aggregation_method,
    )
    ood_features = extract_eigenvalue_sums(
        ood_df,
        timesteps_to_use_final,
        repetitions_to_use,
        aggregation_method=aggregation_method,
    )

    return id_features, ood_features

import subprocess
import sys


"""
Launcher for running calculations on multiple datasets.

Arguments:
    datasets (list): List of datasets with their names and paths.
    num_samples (int): Number of samples to use in each directory.
    outpath (str): Directory where outputs will be saved.
    model_path (str): Path to the directory containing pretrained models.
    model_name (str): Name of the model file to load.
    id_name (str): Name of the in-distribution dataset.
    n_repetitions (int): Number of repetitions to run for each calculation.
    rep_batch (int): Batch size to use during repetitions.
    const (float): Step size for finite difference calculation.
"""


datasets = [
    ("cifar10", "/Data/cifar10-32"),
    ("cifar100", "/Data/cifar100-32"),
]


for name, path in datasets:
    print(f"\nRunning for {name}")
    cmd = [
        sys.executable,  # uses the current conda python
        "eigenscore_main.py",
        "--dataset_path",
        path,
        "--dataset_name",
        name,
        #####Configuration#####
        "--num_samples",
        "20",
        "--outpath",
        "/Test",
        "--model_path",
        "/models/",
        "--model_name",
        "cifar10-32x32.pkl",
        "--id_name",
        "cifar10",
        "--n_repetitions",
        "20",
        "--rep_batch",
        "20",
        "--const",
        "1",
        #######################
    ]
    subprocess.run(cmd)

import pandas as pd
import numpy as np
import os

"""
This script returns the optimal parameters conveniently.
"""

#####Configuration#####
path = "../../GaussianDenoisingPosterior-main/Z_Z_FINAL/Optimized_Parameters"
id_dataset_filters = [
    "celeba",
    "cifar10",
    "cifar100",
    "svhn",
]  # <- your list of desired IDs
#######################


all_files = os.listdir(path)
all_results = []

for each in all_files:
    id_from_filename = each.split("_")[0]
    if id_from_filename not in id_dataset_filters:
        continue  # skip if not matching any desired ID

    each_file = os.path.join(path, each)
    df = pd.read_excel(each_file)
    id = df["ID Dataset"][0]
    ood = df["OOD Dataset"][0]
    timesteps = df["Timestep(s)"]
    method = df["Aggregation"]
    val_auroc = df["Validation_AUROC"]
    test_auroc = df["Test_AUROC"]
    highest_auroc_index = np.array(val_auroc).argmax()
    highest_auroc = val_auroc[highest_auroc_index]

    timesteps = timesteps[highest_auroc_index]
    method = method[highest_auroc_index]
    test_auroc = test_auroc[highest_auroc_index]
    result = {
        "id": id,
        "ood": ood,
        "timesteps": timesteps,
        "method": method,
        "val_auroc": highest_auroc,
        "test_auroc": test_auroc,
    }
    all_results.append(result)

avg = 0
for each in all_results:
    print(each)
    avg += each["test_auroc"]

if all_results:
    print(f"avg_auroc: {avg / len(all_results)}")
else:
    print(f"No results found for ID datasets: {id_dataset_filters}")

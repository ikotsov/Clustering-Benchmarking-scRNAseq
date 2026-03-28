from itertools import product

from src.constants import DATASETS, NORM_METHODS
from src.scripts import run_preprocessing

if __name__ == "__main__":
    total_runs = len(DATASETS) * len(NORM_METHODS)
    print(f"Total runs: {total_runs}")

    run_count = 0
    for dataset, norm_method in product(DATASETS, NORM_METHODS):
        run_count += 1
        print(
            f"[{run_count}/{total_runs}] Preprocessing {dataset} + {norm_method}...")
        try:
            run_preprocessing(
                accession=dataset,
                norm_method=norm_method,
            )
            print()
        except Exception as e:
            print(f"  ⚠ Error: {e}\n")
            continue

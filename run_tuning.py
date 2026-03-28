from itertools import product

from src.constants import DATASETS, NORM_METHODS
from src.tuning.algorithms import ALGORITHM_PARAM_SPECS, run_tuning


ALGORITHMS = tuple(ALGORITHM_PARAM_SPECS.keys())


if __name__ == "__main__":
    total_runs = len(DATASETS) * len(ALGORITHMS) * len(NORM_METHODS)
    print(f"  Total runs: {total_runs}")

    run_count = 0
    for dataset, algorithm, norm_method in product(DATASETS, ALGORITHMS, NORM_METHODS):
        run_count += 1
        print(
            f"[{run_count}/{total_runs}] Tuning {dataset} + {algorithm} + {norm_method}...")
        try:
            run_tuning(
                accession=dataset,
                algorithm=algorithm,
                norm_method=norm_method,
            )
            print()
        except Exception as e:
            print(f"  ⚠ Error: {e}\n")
            continue

    print(f"\n{'='*80}")
    print(f"Tuning complete! Results in:")
    for dataset in DATASETS:
        print(f"  data/{dataset}/outputs/tuning.json")
    print(f"{'='*80}\n")

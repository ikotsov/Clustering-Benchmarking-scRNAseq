"""
Clustering benchmark workflow:

    prepare -> preprocess -> tune -> aggregate_tuning
                          \\-> benchmark -> permutation_test -> aggregate_runs
                          \\-> embed (2D UMAP for plots only)

Every algorithm is benchmarked with its default parameters; the tuned algorithms are benchmarked
a second time with the parameters tuned on the same normalization + PCA representation.
permutation_test scores shuffles of each run's predicted labels, so that results.tsv can compare
every run with random labels of the same cluster sizes (null means and empirical p-values).

Run settings live in config/config.yaml, each dataset's source files in config/datasets.yaml,
and each dataset's analysis parameters in data/datasets/<accession>/config.yaml.

    snakemake --cores 4        # run everything that is missing or out of date
    snakemake -n               # dry run: list the jobs without running them
"""
import shlex

import yaml

from src import paths
from src.dataset_config import load_dataset_config, parse_species


configfile: "config/config.yaml"


with open("config/datasets.yaml") as f:
    SOURCES = yaml.safe_load(f)

DATASETS = config["datasets"]
NORM_METHODS = config["norm_methods"]
ALGORITHMS = config["algorithms"]
TUNED_ALGORITHMS = config["tuning"]["algorithms"]
PCA_OPTIONS = ["pca", "no_pca"]

SCRIPTS = f"{workflow.basedir}/scripts"


# "{norm_method}_{pca}" in the processed file names could otherwise split "log_cpm_no_pca" the wrong way.
wildcard_constraints:
    accession="|".join(DATASETS),
    norm_method="|".join(NORM_METHODS),
    pca="pca|no_pca",
    params="tuned|default",
    algorithm="|".join(ALGORITHMS),


def dataset_species(wildcards):
    return parse_species(load_dataset_config(paths.DATASET_CONFIG.format(accession=wildcards.accession)))


def source_args(wildcards):
    # Each source name becomes an argument of the dataset's prepare script (series_matrix -> --series-matrix).
    return " ".join(
        f"--{name.replace('_', '-')} {shlex.quote(path)}"
        for name, path in SOURCES[wildcards.accession].items()
    )


def benchmark_inputs(wildcards):
    inputs = {
        "data": paths.PROCESSED_DATA.format(**dict(wildcards.items())),
        "config": paths.DATASET_CONFIG.format(accession=wildcards.accession),
        "labels": paths.GROUND_TRUTH_LABELS.format(accession=wildcards.accession),
    }
    if wildcards.params == "default":
        return inputs
    if wildcards.algorithm not in TUNED_ALGORITHMS:
        raise ValueError(f"'{wildcards.algorithm}' is not tuned, so it has no tuned run")
    inputs["params"] = paths.TUNING_RESULT.format(**dict(wildcards.items()))
    return inputs


rule all:
    input:
        expand(paths.RESULTS, accession=DATASETS),
        expand(paths.UMAP, accession=DATASETS, norm_method=NORM_METHODS),
        expand(paths.CLUSTERING_PARAMS, accession=DATASETS) if TUNED_ALGORITHMS else [],


rule prepare:
    input:
        lambda wildcards: list(SOURCES[wildcards.accession].values()),
    output:
        counts=paths.RAW_COUNTS,
        labels=paths.GROUND_TRUTH_LABELS,
    params:
        sources=source_args,
    log:
        "logs/prepare/{accession}.log",
    shell:
        "python {SCRIPTS}/prepare/{wildcards.accession}.py {params.sources}"
        " --out-counts {output.counts} --out-labels {output.labels} 2> {log}"


rule preprocess:
    input:
        raw=paths.RAW_COUNTS,
        config=paths.DATASET_CONFIG,
        annotation=lambda wildcards: paths.GENE_ANNOTATION.format(species=dataset_species(wildcards)),
    output:
        pca=expand(paths.PROCESSED_DATA, pca="pca", allow_missing=True),
        no_pca=expand(paths.PROCESSED_DATA, pca="no_pca", allow_missing=True),
        hvgs=paths.HVGS,
    params:
        pca_variance_ratio=config["pca_variance_ratio"],
    log:
        "logs/preprocess/{accession}/{norm_method}.log",
    shell:
        "python {SCRIPTS}/preprocess.py --raw {input.raw} --config {input.config}"
        " --annotation {input.annotation} --norm-method {wildcards.norm_method}"
        " --pca-variance-ratio {params.pca_variance_ratio}"
        " --out-pca {output.pca} --out-no-pca {output.no_pca} --out-hvgs {output.hvgs} 2> {log}"


rule embed:
    input:
        expand(paths.PROCESSED_DATA, pca="pca", allow_missing=True),
    output:
        paths.UMAP,
    log:
        "logs/embed/{accession}/{norm_method}.log",
    shell:
        "python {SCRIPTS}/embed.py --data {input} --out {output} 2> {log}"


rule tune:
    input:
        data=paths.PROCESSED_DATA,
        config=paths.DATASET_CONFIG,
        labels=paths.GROUND_TRUTH_LABELS,
    output:
        paths.TUNING_RESULT,
    params:
        n_trials=config["tuning"]["n_trials"],
        objective_metric=config["tuning"]["objective_metric"],
        pca_flag=lambda wildcards: "--pca" if wildcards.pca == "pca" else "--no-pca",
    log:
        "logs/tune/{accession}/{norm_method}/{pca}/{algorithm}.log",
    shell:
        "python {SCRIPTS}/tune.py --data {input.data} --config {input.config} --labels {input.labels}"
        " --algorithm {wildcards.algorithm} --norm-method {wildcards.norm_method} {params.pca_flag}"
        " --n-trials {params.n_trials} --objective-metric {params.objective_metric}"
        " --out {output} 2> {log}"


rule aggregate_tuning:
    input:
        expand(
            paths.TUNING_RESULT, norm_method=NORM_METHODS, pca=PCA_OPTIONS, algorithm=TUNED_ALGORITHMS,
            allow_missing=True,
        ),
    output:
        paths.CLUSTERING_PARAMS,
    log:
        "logs/aggregate_tuning/{accession}.log",
    shell:
        "python {SCRIPTS}/aggregate_tuning.py --out {output} {input} 2> {log}"


rule benchmark:
    input:
        unpack(benchmark_inputs),
    output:
        result=paths.RUN_RESULT,
        labels=paths.RUN_LABELS,
    params:
        pca_flag=lambda wildcards: "--pca" if wildcards.pca == "pca" else "--no-pca",
        params_arg=lambda wildcards, input: f"--params {input.params}" if "params" in input.keys() else "",
    log:
        "logs/benchmark/{accession}/{norm_method}/{pca}/{params}/{algorithm}.log",
    shell:
        "python {SCRIPTS}/benchmark.py --dataset {wildcards.accession} --algorithm {wildcards.algorithm}"
        " --norm-method {wildcards.norm_method} {params.pca_flag}"
        " --data {input.data} --config {input.config} --labels {input.labels} {params.params_arg}"
        " --out {output.result} --out-labels {output.labels} 2> {log}"


rule permutation_test:
    input:
        data=paths.PROCESSED_DATA,
        run_labels=paths.RUN_LABELS,
        labels=paths.GROUND_TRUTH_LABELS,
    output:
        paths.RUN_NULL,
    params:
        n_permutations=config["permutation_test"]["n_permutations"],
    log:
        "logs/permutation_test/{accession}/{norm_method}/{pca}/{params}/{algorithm}.log",
    shell:
        "python {SCRIPTS}/permutation_test.py --data {input.data} --run-labels {input.run_labels}"
        " --labels {input.labels} --n-permutations {params.n_permutations}"
        " --out {output} 2> {log}"


def expand_runs(template):
    return expand(
        template, params="default", algorithm=ALGORITHMS,
        norm_method=NORM_METHODS, pca=PCA_OPTIONS, allow_missing=True,
    ) + expand(
        template, params="tuned", algorithm=TUNED_ALGORITHMS,
        norm_method=NORM_METHODS, pca=PCA_OPTIONS, allow_missing=True,
    )


rule aggregate_runs:
    input:
        runs=expand_runs(paths.RUN_RESULT),
        nulls=expand_runs(paths.RUN_NULL),
    output:
        paths.RESULTS,
    log:
        "logs/aggregate_runs/{accession}.log",
    shell:
        "python {SCRIPTS}/aggregate_runs.py --dataset {wildcards.accession} --out {output} {input.runs}"
        " --null-results {input.nulls} 2> {log}"

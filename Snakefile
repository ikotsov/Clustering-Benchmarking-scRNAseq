"""
Clustering benchmark workflow:

    prepare -> preprocess -> tune -> aggregate_tuning
                          \\-> benchmark -> permutation_test ------> aggregate_runs
                                        \\-> biological_evaluation -/
                          \\-> embed (2D UMAP for plots only)
    prepare -> reference_sets -> biological_evaluation
    gene_marker_index -> reference_sets (only for datasets whose config sets marker databases)

Every algorithm is benchmarked with its default parameters; the tuned algorithms are benchmarked
again with the parameters tuned on the same normalization + PCA representation, once per tuning
objective metric: external ones score against the ground truth labels, internal ones the clusters alone,
as without labels. Tuning and run outputs are both labelled "tuned_<metric>" (see src.tuning.objectives.get_params_label).
permutation_test scores shuffles of each run's predicted labels, so that results.tsv can compare
every run with random labels of the same cluster sizes (null means and empirical p-values).
biological_evaluation tests whether the marker genes of each run's clusters match the reference sets
(DE markers of the ground-truth cell types, plus any marker databases) more than expression-matched random
gene sets do; results.tsv gets its summary (bio_* columns).

Run settings live in config/config.yaml, each dataset's source files in config/datasets.yaml,
and each dataset's analysis parameters in data/datasets/<accession>/config.yaml.

    snakemake --cores 4        # run everything that is missing or out of date
    snakemake -n               # dry run: list the jobs without running them
"""
import shlex

import yaml

from typing import get_args

from src import paths
from src.dataset_config import load_dataset_config, parse_marker_databases, parse_species
from src.tuning.objectives import get_params_label, get_tuning_objective
from src.tuning.types import ObjectiveMetric


configfile: "config/config.yaml"


with open("config/datasets.yaml") as f:
    SOURCES = yaml.safe_load(f)

DATASETS = config["datasets"]
NORM_METHODS = config["norm_methods"]
ALGORITHMS = config["algorithms"]
TUNED_ALGORITHMS = config["tuning"]["algorithms"]
for metric in config["tuning"]["objective_metrics"]:
    if metric not in get_args(ObjectiveMetric):
        raise ValueError(f"Unknown tuning objective metric '{metric}' in config/config.yaml")
# The {params} label of each tuning, e.g. "tuned_silhouette", mapped to its metric
TUNED_PARAMS = {get_params_label(metric): metric for metric in config["tuning"]["objective_metrics"]}
PCA_OPTIONS = ["pca", "no_pca"]

SCRIPTS = f"{workflow.basedir}/scripts"


# Wildcards match any text by default, slashes included. pca in particular must stay constrained:
# otherwise processed/{norm_method}/umap.csv.gz would also match processed/{norm_method}/{pca}.csv.gz.
wildcard_constraints:
    accession="|".join(DATASETS),
    norm_method="|".join(NORM_METHODS),
    pca="pca|no_pca",
    params="|".join(["default", *TUNED_PARAMS]),
    algorithm="|".join(ALGORITHMS),


def dataset_species(wildcards):
    return parse_species(load_dataset_config(paths.DATASET_CONFIG.format(accession=wildcards.accession)))


def dataset_annotation(wildcards):
    return paths.GENE_ANNOTATION.format(species=dataset_species(wildcards))


def dataset_marker_index(wildcards):
    # Only datasets whose config sets marker databases need the gene marker index
    config_path = paths.DATASET_CONFIG.format(accession=wildcards.accession)
    return paths.GENE_MARKER_INDEX if parse_marker_databases(load_dataset_config(config_path)) else []


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
    # Tuning and run paths share the {params} label, so a tuned run reads the tuning result of its own label
    inputs["params"] = paths.TUNING_RESULT.format(**dict(wildcards.items()))
    return inputs


rule all:
    input:
        expand(paths.RESULTS, accession=DATASETS),
        expand(paths.RUN_LABELS_TABLE, accession=DATASETS),
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
    benchmark:
        "benchmarks/prepare/{accession}.tsv",
    shell:
        "python {SCRIPTS}/prepare/{wildcards.accession}.py {params.sources}"
        " --out-counts {output.counts} --out-labels {output.labels} 2> {log}"


rule preprocess:
    input:
        raw=paths.RAW_COUNTS,
        config=paths.DATASET_CONFIG,
        annotation=dataset_annotation,
    output:
        pca=expand(paths.PROCESSED_DATA, pca="pca", allow_missing=True),
        no_pca=expand(paths.PROCESSED_DATA, pca="no_pca", allow_missing=True),
        hvgs=paths.HVGS,
    params:
        pca_variance_ratio=config["pca_variance_ratio"],
    log:
        "logs/preprocess/{accession}/{norm_method}.log",
    benchmark:
        "benchmarks/preprocess/{accession}/{norm_method}.tsv",
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
    benchmark:
        "benchmarks/embed/{accession}/{norm_method}.tsv",
    shell:
        "python {SCRIPTS}/embed.py --data {input} --out {output} 2> {log}"


rule tune:
    input:
        data=paths.PROCESSED_DATA,
        config=paths.DATASET_CONFIG,
        # Internal tuning never sees the ground truth labels
        labels=lambda wildcards: (
            paths.GROUND_TRUTH_LABELS if get_tuning_objective(TUNED_PARAMS[wildcards.params]) == "external" else []
        ),
    output:
        paths.TUNING_RESULT,
    params:
        n_trials=config["tuning"]["n_trials"],
        objective_metric=lambda wildcards: TUNED_PARAMS[wildcards.params],
        pca_flag=lambda wildcards: "--pca" if wildcards.pca == "pca" else "--no-pca",
        labels_arg=lambda wildcards, input: f"--labels {input.labels}" if input.labels else "",
    log:
        "logs/tune/{accession}/{norm_method}/{pca}/{params}/{algorithm}.log",
    benchmark:
        "benchmarks/tune/{accession}/{norm_method}/{pca}/{params}/{algorithm}.tsv",
    shell:
        "python {SCRIPTS}/tune.py --data {input.data} --config {input.config} {params.labels_arg}"
        " --algorithm {wildcards.algorithm} --norm-method {wildcards.norm_method} {params.pca_flag}"
        " --n-trials {params.n_trials}"
        " --objective-metric {params.objective_metric}"
        " --out {output} 2> {log}"


rule aggregate_tuning:
    input:
        expand(
            paths.TUNING_RESULT, params=list(TUNED_PARAMS), norm_method=NORM_METHODS, pca=PCA_OPTIONS,
            algorithm=TUNED_ALGORITHMS, allow_missing=True,
        ),
    output:
        paths.CLUSTERING_PARAMS,
    log:
        "logs/aggregate_tuning/{accession}.log",
    benchmark:
        "benchmarks/aggregate_tuning/{accession}.tsv",
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
        params_arg=lambda wildcards, input: (
            f"--params {input.params} --tuning-metric {TUNED_PARAMS[wildcards.params]}"
            if "params" in input.keys() else ""
        ),
    log:
        "logs/benchmark/{accession}/{norm_method}/{pca}/{params}/{algorithm}.log",
    benchmark:
        "benchmarks/benchmark/{accession}/{norm_method}/{pca}/{params}/{algorithm}.tsv",
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
    benchmark:
        "benchmarks/permutation_test/{accession}/{norm_method}/{pca}/{params}/{algorithm}.tsv",
    shell:
        "python {SCRIPTS}/permutation_test.py --data {input.data} --run-labels {input.run_labels}"
        " --labels {input.labels} --n-permutations {params.n_permutations}"
        " --out {output} 2> {log}"


rule gene_marker_index:
    input:
        paths.GENE_MARKER_SOURCES,
    output:
        paths.GENE_MARKER_INDEX,
    log:
        "logs/gene_marker_index.log",
    shell:
        "python {SCRIPTS}/generate_gene_marker_index.py 2> {log}"


rule reference_sets:
    input:
        raw=paths.RAW_COUNTS,
        config=paths.DATASET_CONFIG,
        annotation=dataset_annotation,
        labels=paths.GROUND_TRUTH_LABELS,
        marker_index=dataset_marker_index,
    output:
        paths.REFERENCE_SETS,
    params:
        marker_index_arg=lambda wildcards, input: f"--marker-index {input.marker_index}" if input.marker_index else "",
    log:
        "logs/reference_sets/{accession}.log",
    benchmark:
        "benchmarks/reference_sets/{accession}.tsv",
    shell:
        "python {SCRIPTS}/reference_sets.py --raw {input.raw} --config {input.config}"
        " --annotation {input.annotation} --labels {input.labels} {params.marker_index_arg}"
        " --out {output} 2> {log}"


rule biological_evaluation:
    input:
        raw=paths.RAW_COUNTS,
        config=paths.DATASET_CONFIG,
        annotation=dataset_annotation,
        run_labels=paths.RUN_LABELS,
        reference_sets=paths.REFERENCE_SETS,
    output:
        paths.RUN_BIOLOGICAL,
    params:
        n_samples=config["biological_evaluation"]["n_samples"],
        min_marker_genes=config["biological_evaluation"]["min_marker_genes"],
    log:
        "logs/biological_evaluation/{accession}/{norm_method}/{pca}/{params}/{algorithm}.log",
    benchmark:
        "benchmarks/biological_evaluation/{accession}/{norm_method}/{pca}/{params}/{algorithm}.tsv",
    shell:
        "python {SCRIPTS}/biological_evaluation.py --raw {input.raw} --config {input.config}"
        " --annotation {input.annotation} --run-labels {input.run_labels}"
        " --reference-sets {input.reference_sets} --n-samples {params.n_samples}"
        " --min-marker-genes {params.min_marker_genes} --out {output} 2> {log}"


def expand_runs(template):
    return expand(
        template, params="default", algorithm=ALGORITHMS,
        norm_method=NORM_METHODS, pca=PCA_OPTIONS, allow_missing=True,
    ) + expand(
        template, params=list(TUNED_PARAMS), algorithm=TUNED_ALGORITHMS,
        norm_method=NORM_METHODS, pca=PCA_OPTIONS, allow_missing=True,
    )


rule aggregate_runs:
    input:
        runs=expand_runs(paths.RUN_RESULT),
        nulls=expand_runs(paths.RUN_NULL),
        biologicals=expand_runs(paths.RUN_BIOLOGICAL),
        labels=expand_runs(paths.RUN_LABELS),
    output:
        results=paths.RESULTS,
        labels=paths.RUN_LABELS_TABLE,
    log:
        "logs/aggregate_runs/{accession}.log",
    benchmark:
        "benchmarks/aggregate_runs/{accession}.tsv",
    shell:
        "python {SCRIPTS}/aggregate_runs.py --dataset {wildcards.accession}"
        " --out {output.results} --out-labels {output.labels} {input.runs}"
        " --null-results {input.nulls} --biological-results {input.biologicals}"
        " --run-labels {input.labels} 2> {log}"

# DiReAM code and research data

This package accompanies *DiReAM: Directional Aggregation Before Matching for Compact Defect Localization*. It provides the method implementation, comparison configurations, ordered dataset memberships, saved observations, request measurements and scientific figure inputs.

Start with the [method and configuration guide](docs/METHOD.md), [data and model sources](docs/DATA_SOURCES.md), and [result formats and statistical units](docs/STATISTICS.md). The final figure PDFs are in `figures/`. Manuscript PDFs, typesetting sources and editable artwork are supplied separately.

## Reconstruct saved results on CPU

Use Python 3.12 and the dependencies in `requirements-cpu.txt`: NumPy 2.5.2, Pillow 12.3.0 and SciPy 1.18.1. The five commands below use saved numerical observations; they need no GPU, image dataset or model checkpoint. `ENVIRONMENT.json` records the reference software environment.

Run from this package's root. Choose a new output directory outside the package for each command:

```text
python -I -B code/replay.py verify  --output ../diream-results/verify
python -I -B code/replay.py macros  --output ../diream-results/macros
python -I -B code/replay.py costs   --output ../diream-results/costs
python -I -B code/replay.py budgets --output ../diream-results/budgets
python -I -B code/replay.py export  --output ../diream-results/export
```

| Command | Result |
|---|---|
| `verify` | Checks the supplied inputs and their recorded identities. |
| `macros` | Reconstructs the reported equal-category summaries from per-unit/category observations. |
| `costs` | Reconstructs request median and p95 statistics with the original measurement boundaries. |
| `budgets` | Checks retained-vector counts, feature precision and stored-state payload geometry. |
| `export` | Exports scientific result tables and plotting inputs. |

Inspect the command's output report and exit status. Published table rounding is for presentation; replay uses full-precision observations. Do not edit a checksum or a result to make a check pass.

The additional timing series has a standard-library replay:

```text
python -I -B extensions/timing/replay_timing_extension.py --records extensions/timing/records --output ../diream-results/timing
```

The soybean/terminal analysis has a NumPy scalar replay:

```text
python -I -B extensions/unseen/scripts/recompute_scalar.py --evidence extensions/unseen/records --output ../diream-results/category-transfer
```

See [timing](extensions/timing/README.md) and [same-source category transfer](extensions/unseen/README.md) for their separate cohorts and statistics.

## Work from original images

[INFERENCE.md](docs/INFERENCE.md) explains the image-resource bindings, numerical conventions and fit/predict/evaluate route. Original datasets, checkpoints, large feature caches and reference banks remain external. The supplied selected route targets Windows/CUDA; CPU replay and resource inspection are separate capabilities. The archived experiments ran on GPU, while a second complete GPU rebuild from this distribution has not been completed.

[SOFTWARE_CHECKS.md](docs/SOFTWARE_CHECKS.md) describes implementation tests. [NOTICE.md](NOTICE.md) and [LICENSE_SCOPE.md](LICENSE_SCOPE.md) explain the software grant and the separate terms for data, model weights and upstream material.

## Files

[figure_data_index.json](figure_data_index.json) maps the retained scientific inputs to the article's figure numbers. Figure 1 is a conceptual overview; the index identifies the data-backed main and supplementary figures.

| Directory | Purpose |
|---|---|
| `code/`, `base/code/` | Method, evaluation, saved-result replay and state-loading implementation. |
| `protocol/` | Dataset roles, memberships, configurations, source versions and numerical conventions. |
| `results/`, `source_data/` | Saved observations, full-precision summaries and scientific source records. |
| `extensions/timing/`, `extensions/unseen/` | Separate timing and same-source category-transfer analyses. |
| `figure_sources/shared/data/` | Scientific example photographs, GT, fields and arrays, including negative cases; no layout-production tools are required to read them. |
| `third_party/` | License and NOTICE files for externally acquired upstream implementations. |
| `figures/` | Final article and supplement figure PDFs. |
| `tests/` | Implementation tests and explicitly synthetic fixtures. |

Configuration-by-unit rows, overlapping cohorts, repeated requests and historical reuses are not additional independent efficacy samples. The [statistics guide](docs/STATISTICS.md) defines each denominator and timing window.

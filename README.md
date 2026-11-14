# DiReAM

Research software accompanying **DiReAM: Directional Aggregation Before Matching for Compact Defect Localization**, by **Lin Li and Kang Wang**.

DiReAM inverse-aligns feature grids from four digital quarter-turn readings of a frozen visual encoder, then forms directional means before nearest-normal-reference matching. The selected implementation uses DINOv2 ViT-S/14. The four encoder readings are retained; the aggregation changes the matching stage. See the [method and configuration guide](reproducibility/docs/METHOD.md) for the numerical conventions, reference budgets and comparison profiles.

## Start here

This repository is a **source view of the complete reproducibility package**. It contains the implementation, documentation, source and model identities, license notices, and selected compact protocol files. The saved experimental observations, full membership records, scientific figure inputs and figure PDFs are distributed in the complete archive.

**Cloning this repository alone is not sufficient to run the saved-result replay commands.** First obtain the complete archive and extract it into `reproducibility/` as described below. The supplied replay entry point checks the entire original package before running; a partial source checkout will fail that check by design.

The complete reproduction archive has a **reserved Figshare DOI: [10.6084/m9.figshare.34894161](https://doi.org/10.6084/m9.figshare.34894161)**. The Figshare record is still private; this DOI will become active when the record is published. During peer review, download access is supplied with the manuscript. Use the archive associated with the manuscript version being evaluated; a private review link is not embedded in this public source repository.

## Prepare the complete package

Clone the source repository:

```text
git clone https://github.com/Kane11378/DiReAM.git
cd DiReAM
```

1. Obtain `DiReAM_reproducibility.zip` from the archive access provided with the manuscript.
2. Place the ZIP outside `reproducibility/`. Extract its contents directly into that directory, preserving all paths and bytes. The directory must contain `MANIFEST.json`, `code/`, `protocol/`, `results/` and the other package files without an extra enclosing directory.
3. Keep environments, downloaded external resources and generated outputs outside `reproducibility/`. Do not add files there, change checksums, or edit recorded results to make verification pass.

For example, when the ZIP is next to the repository directory, run this from the repository root:

```text
python -m zipfile -e ../DiReAM_reproducibility.zip reproducibility
```

The tracked files inside `reproducibility/` retain the original archive paths and bytes. Completing that directory from the matching archive preserves the original package-integrity checks. The outer repository README and Git metadata are separate from the checked package.

## Reconstruct saved results on CPU

Use Python 3.12 and an environment outside the checked package. Install the pinned CPU dependencies, then run from `reproducibility/`:

```text
python -m pip install -r reproducibility/requirements-cpu.txt
cd reproducibility
python -I -B code/replay.py verify  --output ../diream-results/verify
python -I -B code/replay.py macros  --output ../diream-results/macros
python -I -B code/replay.py costs   --output ../diream-results/costs
python -I -B code/replay.py budgets --output ../diream-results/budgets
python -I -B code/replay.py export  --output ../diream-results/export
```

Choose new output directories for each run and inspect each command's report and exit status.

| Command | Purpose |
| --- | --- |
| `verify` | Check the supplied inputs and their recorded identities. |
| `macros` | Reconstruct equal-category summaries from saved per-unit and category observations. |
| `costs` | Reconstruct request median and p95 statistics using the recorded measurement boundaries. |
| `budgets` | Check retained-vector counts, precision and stored-state payload geometry. |
| `export` | Export result tables and scientific plotting inputs. |

These commands use saved observations and require no GPU, original image dataset or model checkpoint. They do not regenerate predictions from original images.

The additional timing and same-source category-transfer analyses have separate replay commands, also run from `reproducibility/` after completing the archive:

```text
python -I -B extensions/timing/replay_timing_extension.py --records extensions/timing/records --output ../diream-results/timing
python -I -B extensions/unseen/scripts/recompute_scalar.py --evidence extensions/unseen/records --output ../diream-results/category-transfer
```

See the [timing guide](reproducibility/extensions/timing/README.md), [category-transfer guide](reproducibility/extensions/unseen/README.md) and [statistics guide](reproducibility/docs/STATISTICS.md) for their distinct cohorts, denominators and analysis boundaries.

## Work from original images

The [original-image inference guide](reproducibility/docs/INFERENCE.md) documents resource checks, normal-bank fitting, prediction and evaluation. Follow its resource-binding and validation steps before a full image run.

- Original datasets, model checkpoints, large feature caches and reference banks remain external. Their sources and identities are recorded in the [data and model source guide](reproducibility/docs/DATA_SOURCES.md).
- The selected numerical route targets the recorded Windows/CUDA environment. See `reproducibility/ENVIRONMENT.json` and `reproducibility/requirements-raw-selected.txt`.
- The archived study experiments ran on GPU. A second complete GPU bank-build, prediction and evaluation run from this distribution has **not** been completed.
- Software fixtures, resource checks and saved-result replay have separate purposes. They do not establish an independent reproduction of the full scientific experiments. See [software checks](reproducibility/docs/SOFTWARE_CHECKS.md).

## Find the implementation and records

| Location | Contents |
| --- | --- |
| [`reproducibility/code/`](reproducibility/code/) and [`reproducibility/base/code/`](reproducibility/base/code/) | Method, evaluation, replay and state-loading implementation. |
| [`reproducibility/docs/`](reproducibility/docs/) | Method, data sources, statistics, inference and verification documentation. |
| `reproducibility/protocol/` | Selected compact protocol files in this source view; complete memberships and configurations in the archive. |
| `reproducibility/results/` and `reproducibility/source_data/` | Saved observations and full-precision records supplied in the complete archive. |
| [`reproducibility/figure_data_index.json`](reproducibility/figure_data_index.json) | Mapping from paper figures to scientific inputs supplied in the archive. |
| [`reproducibility/third_party/`](reproducibility/third_party/) | Preserved upstream licenses and notices. |

## Citation and licensing

[CITATION.cff](CITATION.cff) records the software title and authors. The associated work is a manuscript; no journal acceptance or article publication DOI is claimed here. The reserved Figshare DOI above identifies the complete reproduction archive, not the compact GitHub source view. Once the archive is published, cite the verified archive version corresponding to the manuscript results.

The Apache-2.0 grant applies to the DiReAM-authored contributions identified in [`SOFTWARE_LICENSE_FILES.json`](reproducibility/SOFTWARE_LICENSE_FILES.json), under the limits in [`LICENSE_SCOPE.md`](reproducibility/LICENSE_SCOPE.md). Upstream code retains its own terms and attribution. Dataset objects, research observations, model weights and other materials have separate terms; the repository does not apply a blanket software license to them. Read [`NOTICE.md`](reproducibility/NOTICE.md) and the original [`LICENSE`](reproducibility/LICENSE).

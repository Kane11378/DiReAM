# Same-source category transfer

Soybean and terminal are MANTA Tiny categories selected before new query RGB/GT access. According to the recorded data-use audit, neither had previously been processed by this project. Each has 176 fit normals, 64 disjoint calibration normals and all 300 official test objects, with five acquired views per object. All five profiles, shrinkage, endpoints and memberships were fixed before prediction. This tests new categories within the same source, not a new dataset or manufacturing batch.

## Reconstruct saved scalar results

From the package root:

```text
python -I -B extensions/unseen/scripts/recompute_scalar.py --evidence extensions/unseen/records --output ../diream-results/category-transfer
```

Python and NumPy are sufficient. `PER_UNIT.csv` supplies unit AP eligibility, unit AP, native score/threshold, alarms and coverage. The command reconstructs Unit AP, native image AP/AUROC, sensitivity/FAR, coverage/gated coverage and four paired contrasts using the specified seed and 10,000 whole-unit resamples, comparing against `SUMMARY.csv`/`CONTRASTS.csv` at tolerance `1e-12`. Full fields and GT are needed to recompute pooled-pixel AP/AUROC.

## Original-image route

The image pipeline uses the exact DINOv2 S/14 checkpoint, complete specified official source checkout, Tiny agriculture/mechanics archives and the supplied numerical helpers. Resource identities are in `docs/DATA_SOURCES.md`; package versions are in `records/RUNTIME_REQUIREMENTS.json`. FP32 inference, FP64 normal accumulation, disabled TF32, deterministic mode and batching/chunking follow the fixed configuration. Feature caches, banks and full prediction fields are external.

Copy `records/RESOURCE_BINDINGS.example.json` outside the package and bind the actual archives, model and numerical source locations. `study_root` names the study helpers; `reader_root` names the supplied reference modules. A source identity identifies the expected numerical implementation, while a binding locates it on your computer.

From `extensions/unseen/`, with a new external work directory:

```text
python -I -B scripts/run_bound_raw.py --bindings ../../../diream-resources/UNSEEN_RESOURCE_BINDINGS.json --records records --workdir ../../../diream-category-run --stage check-bindings
python -I -B scripts/run_bound_raw.py --bindings ../../../diream-resources/UNSEEN_RESOURCE_BINDINGS.json --records records --workdir ../../../diream-category-run --stage stage-rgb
python -I -B scripts/run_bound_raw.py --bindings ../../../diream-resources/UNSEEN_RESOURCE_BINDINGS.json --records records --workdir ../../../diream-category-run --stage encode --category soybean
python -I -B scripts/run_bound_raw.py --bindings ../../../diream-resources/UNSEEN_RESOURCE_BINDINGS.json --records records --workdir ../../../diream-category-run --stage fit --category soybean
python -I -B scripts/run_bound_raw.py --bindings ../../../diream-resources/UNSEEN_RESOURCE_BINDINGS.json --records records --workdir ../../../diream-category-run --stage predict --category soybean
```

Repeat encode/fit/predict for terminal, then evaluate only after both categories' five-profile predictions are complete. The wrapper preserves incomplete outputs for recovery and validates actual completed payloads before reuse; a progress marker alone does not establish array completion.

The original category experiment completed on GPU. Recorded CPU checks covered source imports, strict checkpoint/model construction with zero forward calls, generated-array contracts and one existing normal-object regression. They are implementation checks, not new efficacy samples or a second complete raw GPU reproduction. The distribution has not had that second full GPU run.

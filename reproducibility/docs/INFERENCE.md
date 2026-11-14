# Original-image inference and bank reconstruction

The selected portable image route covers all 12 VisA categories and native paddy. It rebuilds normal banks from original RGB and the supplied small normal-metric matrices, predicts the declared calibration/query roles, and evaluates annotations after prediction validation. Earlier diagnostics remain available through their saved results and configurations.

Use the checkpoints, dataset members and source commits in [DATA_SOURCES.md](DATA_SOURCES.md). The numerical backend targets the recorded Windows/CUDA environment in `ENVIRONMENT.json` and `requirements-raw-selected.txt`: FP32 encoding, FP64 mean/transform accumulation, deterministic operations and disabled TF32. A smaller backbone, reduced reference count, lower precision or approximate search defines a different configuration.

## Inspect the selected configuration

```text
python -I -B code/inference.py plan
python -I -B code/inference.py check-resources --category chewinggum --profile M_W_FC4 --budget B --bindings ../diream-resources/RESOURCE_BINDINGS.json
```

`../diream-resources/RESOURCE_BINDINGS.json` is your filled external copy of `RESOURCE_BINDINGS.example.json`. The resource check identifies missing paths or changed checkpoint bytes; execution additionally checks the actual dataset members. Only category/profile/budget combinations in the plan are supported. Exact K is not clipped. Paddy has five camera-slot banks; a pooled native bank receives their combined budget once.

## Check resources before model construction

```text
python -I -B code/public_raw_preflight.py --reader . --category chewinggum --profile M_W_FC4 --budget B --bindings ../diream-resources/RESOURCE_BINDINGS.json --members all --device cpu --probe-runtime --output ../diream-checks/RESOURCES.json
```

This checks all selected members, their source bytes/crops, disjoint roles, source/model identities, normal matrices and K without model or annotation-pixel access. `--members sample` is a small resource/decode check, not a replacement for checking the full role. Original checkpoint and archive hashes identify the study payloads; they are not assertions that their providers published those hashes.

## Build the run checks and execute

The checks below test the selected software, CUDA machine and a small generated/integration case before a full bank build. Keep their output outside the package. The integration case uses a reduced reference count and K=64 for software testing only; scientific `fit` retains the full configured reference role and K.

```text
python -I -B code/public_rebuild_evidence.py software --reader . --output ../diream-checks/software
python -I -B code/public_rebuild_evidence.py machine --reader . --output ../diream-checks/machine
python -I -B code/public_rebuild_evidence.py integration --reader . --resources ../diream-checks/RESOURCES.json --software ../diream-checks/software/SOFTWARE.json --machine ../diream-checks/machine/MACHINE.json --bindings ../diream-resources/RESOURCE_BINDINGS.json --output ../diream-checks/integration
python -I -B code/public_rebuild_evidence.py assemble --reader . --resources ../diream-checks/RESOURCES.json --software ../diream-checks/software/SOFTWARE.json --machine ../diream-checks/machine/MACHINE.json --integration ../diream-checks/integration/INTEGRATION.json --output ../diream-checks/bundle
python -I -B code/public_rebuild_evidence.py validate --reader . --bundle ../diream-checks/bundle/REBUILD_EVIDENCE.json
python -I -B code/inference.py fit --category chewinggum --profile M_W_FC4 --budget B --bindings ../diream-resources/RESOURCE_BINDINGS.json --rebuild-evidence ../diream-checks/bundle/REBUILD_EVIDENCE.json --output ../diream-image-run
python -I -B code/inference.py predict --category chewinggum --profile M_W_FC4 --budget B --bindings ../diream-resources/RESOURCE_BINDINGS.json --rebuild-evidence ../diream-checks/bundle/REBUILD_EVIDENCE.json --output ../diream-image-run
python -I -B code/inference.py evaluate --category chewinggum --profile M_W_FC4 --budget B --bindings ../diream-resources/RESOURCE_BINDINGS.json --rebuild-evidence ../diream-checks/bundle/REBUILD_EVIDENCE.json --output ../diream-image-run
```

Use the same category, profile, budget, bindings and run directory through fit/predict/evaluate. Prediction preserves the complete ordered calibration/query roles; evaluation checks completion before opening query masks. Resume only artifacts whose actual payloads and state bindings validate. Resource/machine checks must describe the machine and data used for that run.

The supplied small normal transform is a numerical input to this route. Rebuilding it from original fit-normal responses is a separate operation using the retained normal-statistics kernels. Matching a reconstructed state's structure does not by itself prove equality to an old omitted bank or prediction field. Exact old-state reuse needs its matching payloads and regressions.

## Scope

The archived study ran image inference on GPU. The distributed route has CPU resource/software checks, but a second full CUDA bank-build/prediction/evaluation run from this distribution has not been completed. Full datasets, checkpoints, banks and spatial predictions are external. The five CPU replay commands start from saved observations and do not perform this image pipeline. New-device timing is a new implementation-specific measurement series, not a rerun of the supplied request CSVs.

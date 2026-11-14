# Supplemental request measurements

This analysis measures five deliberately selected, previously seen VisA categories: PCB2, PCB3, PCB4 and macaroni2 include reported mean-matching losses; chewinggum is a historical timing anchor. It is not an unbiased category sample or an independent efficacy validation.

Five fixed profiles use the original banks, indices and normal transform: N_W_F0, I_W_U, M_W_U, I_W_FC4 and M_W_FC4. There are three fresh process blocks per category/profile, 75 process launches, 2,700 resident requests and 3,000 total requests. Each process uses the same 18 fixed inputs: six calibration normals, six test normals and six test anomalies, with one first request, three warmups and two cycles of its fixed permutation. Numeric OS PIDs were reused; process-launch identity, not the PID alone, defines a block.

## Reconstruct the saved measurements

From the package root:

```text
python -I -B extensions/timing/replay_timing_extension.py --records extensions/timing/records --output ../diream-results/timing
```

Only Python's standard library is needed. The replay reconstructs per-process median and linear-interpolation p95, the median/range across three process medians, and paired-block mean-versus-independent reduction percentages. Outputs must be outside the package. The supplied entry point reads saved CSV/JSON observations; it does not launch new GPU timing.

`ALL_REQUESTS.csv` records request phase, input identity/role, encoder/PCA calls and total request seconds. `PROCESS_SUMMARIES.csv` gives initialization/first-request costs, resident median/p95 in milliseconds, memory in bytes and the historical quality/state binding. The quality point comes from the corresponding full-category evaluation, not from the 18 timing inputs. A p95 and a three-process range are descriptive, not confidence intervals.

## Measurement boundary

Complete resident requests include file read/RGB decode, required FP32 encoder readings, raw-token PCA foreground where required, transforms, retrieval, projection/smoothing and native score. Imports, model/state loading, input identity checks, validation, output hashing and persistence lie outside the timer. Input hashing can warm the OS file cache; no cold-cache result is claimed. No query features or foreground masks are cached. CUDA synchronization bounds the request timer.

The measurement environment was Windows, RTX 5090, PyTorch 2.13.0+cu130, CUDA runtime 13.0 and four Torch CPU threads, with deterministic operations and TF32 disabled. The fixed source, checkpoint, input and state identities accompany the observations. Numerical regressions checked the timed implementations against the saved fields; no model fitting, query-mask access, AP calculation or parameter selection occurred in this timing analysis.

The separate eight-process stage diagnostic uses PCB2/chewinggum, I/M and U/FC4 with three inputs and nine resident requests per process. Extra synchronization instruments the stages, so its means/shares do not decompose or replace the main uninstrumented median. Earlier request series retain their own boundaries and must not be pooled into these ratios. A new hardware/source measurement is a separate series.

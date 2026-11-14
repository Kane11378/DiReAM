# Result formats and statistical units

Saved observations are the inputs to replay. Rounded paper values are presentation outputs. Keep category, configuration, fitted-state identity and observation origin attached to each result.

## Efficacy observations

`results/evaluation/RESULTS_CATEGORY.csv` contains 113 category evaluations, including 87 new computations and 26 historical reuses. `RESULTS_UNITS.csv` contains 19,560 configuration-by-unit rows; this is not a count of independent objects. Overlapping cohorts and historical reuses do not add independent samples.

| Field or quantity | Reader meaning |
|---|---|
| `category`, `unit_id` / `global_identity` | Dataset category and original inspected-object identity. |
| `profile_id` / `profile`, `budget_id` | The declared method configuration and reference-vector budget; see METHOD.md and the configuration records. |
| `state_sha256` | Identity of the fitted normal state used by that result; it is not the digest of a repackaged metadata JSON. |
| `actual_views` | Genuinely acquired views belonging to the inspected object; rotations are not counted as views. |
| `ap_eligible`, `Unit_AP_eligible` | Whether a unit has positive-mask support in the stated scoring domain and contributes to its Unit AP average. |
| `unit_ap`, `Unit_AP` | Within-unit pixel average precision, jointly ranking eligible positions over the acquired views. |
| `pixel_ap`, `Pixel_AP`; pixel AUROC | Category-pooled spatial endpoints. They cannot be reconstructed by averaging unit AP scalars. |
| `native_score`, image AP/AUROC | The detector's own object score and its ranking of normal/anomalous objects. |
| `normal_alarm_threshold`, `native_threshold`, `threshold` | Threshold determined from the stated normal calibration role. |
| `alarmed`, `alarm`, `sensitivity`, `FAR` | Alarm decision, anomalous-object sensitivity and normal-object false-alarm rate for that threshold. |
| `coverage_01`, `coverage_0.01` (and 0.05) | The reported position-budget coverage endpoint at 1% (or 5%). |
| `gated_coverage_*` | Coverage after the object's alarm decision; preserve the corresponding threshold and denominator. |

AP/AUROC and coverage values use fractions unless a field explicitly identifies percent or percentage points. A difference ending in `_pp` is already in percentage points. A zero is not the same as an ineligible or uncomputed observation. Preserve the supplied eligibility and status/reason fields.

The main paired analysis uses 10,000 whole-unit paired draws, seed 20261022, within fixed categories. Its 69 declared contrasts share 68 logical estimates. Categories, fitted normal states and selected profiles stay fixed. Intervals are conditional pointwise 95% intervals, without multiplicity correction or category-population resampling. The earlier 2,000-draw analysis has its own ordering and must remain separate.

## Request measurements

The main common-request series contains 25 configurations: one first request, two warmups and 27 resident requests per configuration, 750 requests total. Median and linear-interpolation p95 use resident requests. Quality is paired with cost only for the same category/profile/state.

`resident` CUDA peak includes initialization/model and state loading through first, warmup and resident requests. RGB read/decode is timed; preverification, result validation, hashing and persistence are outside the request timer. The earlier `resident_warm` series has a different peak/timing boundary and is not interchangeable.

The additional timing analysis uses 75 process launches and 2,700 resident requests. See `extensions/timing/README.md` for block pairing and the separate instrumented stage diagnostic. Repeated requests are cost observations, not additional efficacy samples. A request p95 or range across three process medians is descriptive, not a confidence interval.

## Reference storage

`STATE_BUDGETS.csv` distinguishes candidates from retained vectors, slot-level from pooled rows, FP32 feature bytes from metric/center/search/index/serialized payloads, and host storage from device feature payload. Feature bytes exclude encoder weights, allocator overhead and process RSS. Preserve `row_level`, slot, dtype, state identity and scope when summing rows; do not add different storage scopes into one footprint.

The following are actual columns of that file, not a suggested new schema:

| Column | Meaning and accounting rule |
|---|---|
| `run_id`, `cohort_id`, `category`, `profile_id`, `budget_id` | Identifiers joining the row to its reported run, cohort and configuration; they are not additional observations. |
| `state_id`, `state_sha256` | Logical and byte identities of the fitted state used in the reported run. A path-edited public metadata file is not the original state preimage. |
| `row_level`, `slot_id` | Whether this physical bank is a camera-slot bank or a pooled bank, and the corresponding slot identity. Actual `row_level` values are `slot` and `pooled`. |
| `candidate_rows`, `retained_rows`, `feature_dim`, `feature_dtype` | Candidate vectors before selection, stored vectors after selection, vector dimension and numeric storage type. The supplied bank rows use 384-dimensional `float32` vectors. |
| `feature_bytes` | Stored feature-array payload; checked as retained rows × 384 × 4 bytes. |
| `metric_center_bytes` | Payload of the fitted normal metric/centering arrays, separate from the feature bank. |
| `search_aux_bytes` | Additional numerical arrays used by the recorded search implementation, separate from feature and normal-metric payloads. |
| `index_metadata_bytes` | Selected-original-row index payload; the replay checks 8 bytes per retained vector. |
| `state_metadata_bytes` | Recorded inference-state metadata-file bytes. This describes the saved state, not the size of a rewritten public metadata document. |
| `serialized_bytes` | Recorded bank, fitted-normal-model and index files, with shared inference-state JSON charged once. The normal model here is not the pretrained encoder. |
| `serialized_scope` | States that the bank/model/index files are counted and shared inference-state JSON is assigned only to the first row. It defines which files the preceding byte count covers. |
| `sum_policy` | States that each physical state/slot or pooled bank appears once and that no separate total row should be added. It prevents double-counting a shared bank. |
| `profiles_served`, `slots_served` | Semicolon-separated profile/slot identities served by this bank. Multiple consumers do not multiply its stored bytes. |
| `bank_sha256`, `indices_sha256`, `model_sha256` | Original bank, selected-index and fitted-normal-model payload identities. They identify numerical inputs, not a current folder name. |
| `device_feature_bytes` | Feature-array payload used on the device; it is not total device memory, a CUDA allocation peak or process RSS. |
| `device_payload_status` | Records whether per-bank geometry was checked against the measured loaded-state total. An empty cell supplies no such per-bank assertion; it does not mean zero device memory. |
| `status` | The recorded verification state of this row, not an efficacy outcome. |
| `source_kind` | Bank-construction provenance. Actual values distinguish a state fitted for its recorded run (`CURRENT_RUN_FIT_STATE`), a historically identified state without prediction reuse (`HISTORICAL_STATE_IDENTITY_VERIFIED_NOT_PREDICTION_REUSE`), a fit after re-encoding RGB (`RAW_RGB_REENCODED_FIT`), and a fit from checked cached features (`VERIFIED_CACHE_FIT`). |

The category/unit result tables use their own `source_kind` values: `RAW_RECOMPUTED` for recomputed predictions and `HISTORICAL_REUSED_VERIFIED` for checked historical outputs. In their `execution_role` column, `planned_quality` denotes the declared quality evaluation; `timing_support_quality` denotes a quality record used to bind a timing configuration. Neither role creates another independently sampled cohort. The category strata `prior_used6` and `remaining6` retain the six-category use-history grouping; the latter is not a newly held-out validation set. `status` and `reason` describe record validity/eligibility; for example, `normal_unit` is distinct from `eligible_abnormal_positive_GT`.

The soybean/terminal scalar replay reconstructs Unit AP, native image endpoints, alarms, coverage and its paired contrasts. The supplied Pixel AP/AUROC values require full fields and GT for an independent recomputation; unit scalars alone do not recreate those spatial rankings.

# Method and configurations

DiReAM uses four quarter-turn responses of a frozen visual encoder. Feature grids are inverse-aligned to the same image locations. Query means are compared with normal-reference means before nearest-reference matching, using the reported normal-data geometry and reference budget.

The selected DINOv2 ViT-S/14 path uses the checkpoint without registers, 448-square views and 32×32 token grids. Each directional component is unit-normalized. Means accumulate in FP64 and are then cast to FP32 without a second normalization. The stored normal transform is applied in the recorded order. Exact-K representative selection retains seed 0, projection dimension 128, ten starting points and the upstream tie/duplicate behavior.

## Read configuration names

| Label | Meaning |
|---|---|
| `N` | Single query reading. |
| `I` | Four separately matched directional readings, combined after matching. |
| `M` | Directional means formed before matching. |
| `W` | Within-anchor normal directional geometry. |
| `U` | No foreground weighting. |
| `F0` | Foreground weighting from direction 0. |
| `FC4` | The declared four-turn foreground-mask combination and ordering. |
| `B`, `half_B`, `double_B` | The category-specific reference-vector budget and its declared variants. |

N and I share selected component vectors in the shared-state controls. Mean matching uses its own bank of normal means at the specified budget. Ordinary covariance, raw means and the complete competing detectors remain separate comparisons. The public profile ID must be read with its category, budget, foreground policy and fitted-state identity; a short label alone does not identify an experiment.

Paddy has five genuinely acquired views with separate camera-slot states. Digital rotations are additional readings of a view, not additional acquired views. The native no-normal-rotation comparator uses direct normal images and one query reading per acquired view. Pooled native support receives the combined slot budget once.

The fixed memberships and configurations in `protocol/` define the exact cohorts. The main VisA follow-up includes previously used query identities. Soybean and terminal were selected before their new query RGB/GT access, but are categories from the same MANTA source rather than a new dataset or acquisition batch. Earlier diagnostics keep their own cohorts and exploratory roles.

See the accompanying article and supplement for equations and interpretation. This guide explains executable labels; it does not replace their stated comparison boundaries.

## Choosing a configuration

Reference storage and query time constrain different parts of the detector. A single reading may be adequate when response time is the priority. Mean matching is a candidate when additional readings improve localization but independent matching costs too much. The article's SI Section S14.6 interprets these measured choices; its quality and time thresholds are illustrative, not established production requirements.

For PCB3, five controls have 50.46 MiB reference-feature banks. With illustrative requirements of at least 51% Unit AP and an 85 ms typical-time limit, the four-mask mean profile meets both at 51.67% and 80.0 ms. These are point summaries from [process measurements](../extensions/timing/records/PROCESS_SUMMARIES.csv), with the [individual requests](../extensions/timing/records/ALL_REQUESTS.csv) retained. The three process medians range from 66.8 to 89.7 ms. These summaries describe typical resident-request time; a tail-latency requirement uses the corresponding request quantiles. The example compares the five controls in the same repeated-request series.

The [budget and cost records](../results/evaluation/TIMING_SUMMARY.csv) also retain paddy's half-budget mean bank: 26.40 MiB, 59.29% Unit AP and 217.0 ms in its separate request series. Compare costs within their respective measurement series and timing boundaries. Bank payload excludes the encoder, auxiliary state and temporary tensors. Unit AP measures spatial ranking within anomalous objects; use the supplied coverage and alarm records when those are the intended inspection outcomes. Choose a configuration on validation data before testing a deployment rule on held-out queries.


"""Whole-unit statistics with tie-grouped AP, AUROC and paired resampling.

The numerical functions originate from prospective_statistics_v1.py, SHA256
57d4a95bc7833f3d020de9d902c18e1e5a6c97a8bbccaf580b14b4673dc1f6df.
Callers provide arrays and validated data bindings; this module has no image,
model or dataset-discovery interface.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import csv,hashlib,json,math
import numpy as np

SEED = 10220261017


REPLICATES = 2000


PDN_ARMS = ("single", "normalaug", "independent", "aligned", "cyclic", "querymin", "meanfeature", "sourcebound")


DINO_ARMS = ("single", "normalaug", "independent", "cyclic", "meanfeature", "meanfeature_unit")


DEFAULT_SYSTEMS = tuple("pdn_"+a for a in PDN_ARMS) + tuple("dino_"+a for a in DINO_ARMS) + (
    "patchcore_base", "patchcore_c4", "anomalydino_base", "anomalydino_c4") + tuple("pdn_"+a+"_raw_secondary" for a in PDN_ARMS)


DEFAULT_COMPARISONS = (
    ("pdn_independent", "pdn_normalaug"), ("pdn_cyclic", "pdn_independent"),
    ("pdn_sourcebound", "pdn_independent"), ("pdn_sourcebound", "pdn_cyclic"),
    ("pdn_meanfeature", "pdn_independent"), ("pdn_independent", "patchcore_base"),
    ("pdn_independent", "anomalydino_base"), ("patchcore_c4", "patchcore_base"),
    ("anomalydino_c4", "anomalydino_base"), ("dino_independent", "dino_normalaug"),
    ("dino_cyclic", "dino_independent"), ("dino_meanfeature", "dino_independent"),
    ("dino_meanfeature_unit", "dino_independent"),
    ("pdn_independent_raw_secondary", "pdn_normalaug_raw_secondary"))


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text("utf-8-sig"))


def write(path, obj): Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def csv_write(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def binary_metrics(labels, scores):
    """Tie-grouped noninterpolated AP and AUROC; undefined values stay None."""
    y, s = np.asarray(labels, bool).ravel(), np.asarray(scores).ravel()
    require(y.shape == s.shape and np.isfinite(s).all(), "Invalid binary metric input")
    positives, negatives = int(y.sum()), int((~y).sum())
    if not len(y): return dict(AP=None, AUROC=None, positives=0, negatives=0)
    order = np.argsort(-s, kind="stable"); yy, ss = y[order], s[order]
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss)-1]
    tp = np.cumsum(yy, dtype=np.int64)[ends]; fp = ends+1-tp
    dt, df = np.diff(np.r_[0, tp]), np.diff(np.r_[0, fp])
    AP = float(np.sum(dt/(positives or 1)*tp/(ends+1))) if positives else None
    AUC = float(np.sum(dt*(negatives-fp+.5*df))/(positives*negatives)) if positives and negatives else None
    return dict(AP=AP, AUROC=AUC, positives=positives, negatives=negatives)


def bootstrap_binary_metrics(labels, scores, object_draws):
    """Whole-object label-stratified draws; vectorized weighted tie groups."""
    y, s, draws = np.asarray(labels, bool), np.asarray(scores), np.asarray(object_draws)
    b, n = draws.shape
    require(n == len(y) == len(s), "Detection bootstrap draw geometry changed")
    require(n > 0 and np.isfinite(s).all(), "Empty/nonfinite object scores")
    w = np.zeros((b, n), np.int32)
    np.add.at(w, (np.arange(b)[:, None], draws), 1)
    order = np.argsort(-s, kind="stable"); yy, ss, ww = y[order], s[order], w[:, order]
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1])+1]
    dt = np.add.reduceat(ww*yy, starts, axis=1).astype(np.float64)
    df = np.add.reduceat(ww*(~yy), starts, axis=1).astype(np.float64)
    tp, fp = np.cumsum(dt, 1), np.cumsum(df, 1)
    p, nneg = tp[:, -1], fp[:, -1]
    precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)>0)
    a = np.sum(dt*precision, 1)/p if np.all(p > 0) else None
    u = np.sum(dt*(nneg[:, None]-fp+.5*df), 1)/(p*nneg) if np.all((p > 0)&(nneg > 0)) else None
    return dict(AP=a, AUROC=u)


def interval(values, coverage=.95):
    if values is None: return None
    a = np.asarray(values, np.float64)
    require(a.ndim == 1 and np.isfinite(a).all(), "Nonfinite bootstrap statistic")
    lo, hi = np.quantile(a, [(1-coverage)/2, (1+coverage)/2], method="linear")
    return dict(coverage=coverage, lower=float(lo), upper=float(hi), method="paired whole-object percentile bootstrap")


@dataclass(frozen=True)
class StatisticsConfig:
    source_categories: tuple = (("MANTA", ("paddy", "maize", "square_button_cap")),
                                ("VisA", ("cashew", "capsules", "pipe_fryum", "pcb1")))
    expected_systems: tuple = DEFAULT_SYSTEMS
    comparisons: tuple = DEFAULT_COMPARISONS
    primary_pair: tuple = ("pdn_independent", "pdn_normalaug")
    seed: int = SEED
    replicates: int = REPLICATES
    expected_hw: tuple = (256, 256)
    budgets: tuple = (.01, .05)

    def as_json(self):
        return dict(source_categories=[dict(source=s, categories=list(c)) for s,c in self.source_categories],
                    expected_systems=list(self.expected_systems), comparisons=[list(p) for p in self.comparisons],
                    primary_pair=list(self.primary_pair), primary_endpoint="mean_object_pixel_AP",
                    primary_multiplicity="two fixed source macros; two-sided97.5% Bonferroni intervals plus95%; no category selection",
                    secondary_intervals="pointwise95%, descriptive; no familywise or false-discovery claim",
                    seed=self.seed, replicates=self.replicates, expected_hw=list(self.expected_hw), budgets=list(self.budgets),
                    resampling="one sequential default_rng(seed), frozen source/category order; per group draw AP-eligible anomalous objects, normal objects, then detection normal and anomalous label strata; reuse draws across all systems/endpoints",
                    inference_scope="conditional on fixed reference and healthy-calibration libraries; classes fixed, no view/pixel/category bootstrap",
                    missing_GT_rule="retain every object label; invalid pixel GT excluded only from pixel metrics; valid zero-positive anomalies retained in pooled pixel and object endpoints, excluded from per-object AP/coverage with null and explicit counts",
                    pooled_pixel_intervals="pooled pixel AP/AUROC point estimates only; no pixel-level CI",
                    map_threshold_rule="separately frozen calibration q99 higher, strict >; nominal calibration quantile, no population1% guarantee",
                    calibration_counts_policy="nominal64 per category; actual nonempty reviewed ready/keep subset after frozen invalid/duplicate exclusions, no replacement or arbitrary reduction; threshold file and actual field IDs must exactly match that subset; actual n reported",
                    budget_rule="per object floor(fraction*V*H*W), all real views jointly ranked, stable ties C-order slot/y/x",
                    raw_PDN="all eight raw fields projected to256 are predeclared secondary; no per-material raw/calibrated winner selection")


@dataclass
class Prediction:
    name: str
    source: str
    category: str
    family: str
    arm: str
    object_ids: tuple
    load: Callable[[str], tuple]
    map_threshold: float | None
    object_threshold: float | None
    provenance: dict = field(default_factory=dict)
    pins: tuple = ()
    prediction_seal: str | None = None


def array_prediction(name, source, category, records, maps, object_scores, *, family="fixture", arm="fixture", map_threshold=None, object_threshold=None):
    """Explicit caller arrays only, used by CPU synthetic fixtures."""
    ids = tuple(r["global_identity"] for r in records); lookup = {k:i for i,k in enumerate(ids)}
    maps, scores = np.asarray(maps), np.asarray(object_scores)
    require(len(maps) == len(scores) == len(ids), "Prediction array count changed")
    return Prediction(name, source, category, family, arm, ids,
                      lambda identity: (maps[lookup[identity]], float(scores[lookup[identity]])),
                      map_threshold, object_threshold, dict(adapter="explicit caller arrays, synthetic fixture only"))


def validate_group(gt, records, source, category, config):
    gt = np.asarray(gt)
    require(gt.dtype == bool and gt.ndim == 4 and gt.shape[-2:] == config.expected_hw, "Explicit bool[N,V,H,W] GT required")
    require(len(gt) == len(records) and len(gt) > 0 and gt.shape[1] > 0, "GT object/real-view count changed")
    ids = [r["global_identity"] for r in records]
    require(len(set(ids)) == len(ids), "Duplicate GT object identity")
    require(all(r["source"] == source and r["category"] == category for r in records), "GT source/category identity changed")
    require(all(isinstance(r["object_anomalous"],(bool,np.bool_)) and isinstance(r["valid_pixel_GT"],(bool,np.bool_)) for r in records), "GT/object validity labels must be explicit booleans")
    bad = np.array([r["object_anomalous"] for r in records], bool)
    valid = np.array([r["valid_pixel_GT"] for r in records], bool)
    counts = gt.reshape(len(gt), -1).sum(1)
    require(not np.any(counts[~bad & valid]), "Normal metadata has positive GT")
    for i,r in enumerate(records):
        if valid[i] and "processed_positive_count" in r:
            require(r["processed_positive_count"] == int(counts[i]), "Processed GT count changed")
    eligible = bad & valid & (counts > 0)
    return bad, valid, counts, eligible


def make_plans(groups, config):
    rng = np.random.default_rng(config.seed); plans = {}; exported = {}
    for number,(source, cats) in enumerate(config.source_categories):
        for ci,cat in enumerate(cats):
            gt, rec = groups[(source,cat)]
            bad, valid, counts, eligible = validate_group(gt, rec, source, cat, config)
            normals, anomalies, apids = np.flatnonzero(~bad), np.flatnonzero(bad), np.flatnonzero(eligible)
            n, b = len(rec), config.replicates
            ad = rng.integers(0,len(apids),(b,len(apids)),dtype=np.int32) if len(apids) else np.empty((b,0),np.int32)
            nd = rng.integers(0,len(normals),(b,len(normals)),dtype=np.int32) if len(normals) else np.empty((b,0),np.int32)
            # Draw the two original label strata separately, including empty/invalid GT.
            detn = normals[rng.integers(0,len(normals),(b,len(normals)))] if len(normals) else np.empty((b,0),np.int32)
            deta = anomalies[rng.integers(0,len(anomalies),(b,len(anomalies)))] if len(anomalies) else np.empty((b,0),np.int32)
            det = np.concatenate([detn,deta],1).astype(np.int32)
            plans[(source,cat)] = dict(AP=ad, normal=nd, detection=det, AP_ids=apids, normal_ids=normals,
                                       bad=bad, valid=valid, positive_counts=counts, eligible=eligible)
            key = f"g{number}_{ci}"
            exported[key+"_AP"] = ad; exported[key+"_normal"] = nd; exported[key+"_detection"] = det
    return plans, exported


def _mean_and_boot(values, draws):
    values = np.asarray(values, np.float64)
    return (float(values.mean()), values[draws].mean(1)) if len(values) else (None,None)


def evaluate_one(pred, gt, records, plan, config):
    ids = [r["global_identity"] for r in records]
    require(set(pred.object_ids) == set(ids) and len(pred.object_ids) == len(ids), "Prediction/GT exact identity coverage differs")
    require(all(x is None or np.isfinite(x) for x in [pred.map_threshold,pred.object_threshold]), "Nonfinite fixed threshold")
    n = len(records); bad, valid, positive = plan["bad"],plan["valid"],plan["positive_counts"]
    maps = np.empty(gt.shape, np.float32); scores = np.empty(n, np.float64)
    rows = []; budget_keys = []
    for f in config.budgets: budget_keys += [f"coverage_{f:g}",f"hit_{f:g}"]
    for i,r in enumerate(records):
        a,s = pred.load(r["global_identity"]); a = np.asarray(a)
        require(a.shape == gt[i].shape and np.isfinite(a).all() and np.isfinite(s), "Prediction geometry/nonfinite map or score")
        require(a.dtype == np.float32, "Actual stored standardized map must be FP32; no silent score conversion")
        maps[i], scores[i] = a,s
        row = dict(source=pred.source, category=pred.category, system=pred.name, family=pred.family, arm=pred.arm,
                   index=i, global_identity=r["global_identity"], object_anomalous=bool(bad[i]), valid_pixel_GT=bool(valid[i]),
                   native_positive_count=r.get("native_positive_count"), processed_positive_count=int(positive[i]) if valid[i] else None,
                   nominal_calibration_objects=pred.provenance.get("nominal_calibration_objects"),
                   actual_calibration_objects=pred.provenance.get("actual_calibration_objects"),
                   empty_processed_anomalous=bool(bad[i] and valid[i] and not positive[i]),
                   pixel_AP_eligible=bool(plan["eligible"][i]), object_pixel_AP=None,
                   native_object_score=float(s), map_threshold=pred.map_threshold, native_object_threshold=pred.object_threshold,
                   normal_false_area=None, normal_map_any_alarm=None, normal_native_object_alarm=None)
        row.update({k:None for k in budget_keys})
        if plan["eligible"][i]:
            row["object_pixel_AP"] = binary_metrics(gt[i],a)["AP"]
            flat_y = gt[i].ravel(); order = np.argsort(-a.ravel(),kind="stable"); tp = np.cumsum(flat_y[order])
            for f in config.budgets:
                count = int(np.floor(f*a.size)); covered = int(tp[count-1]) if count else 0
                row[f"budget_positions_{f:g}"] = count
                row[f"covered_positive_{f:g}"] = covered
                row[f"coverage_{f:g}"] = covered/int(positive[i]); row[f"hit_{f:g}"] = int(covered > 0)
        if not bad[i]:
            if pred.map_threshold is not None:
                alarm = a > pred.map_threshold
                row["normal_false_area"] = float(alarm.mean()); row["normal_map_any_alarm"] = int(alarm.any())
                row["normal_false_positions"] = int(alarm.sum()); row["normal_total_positions"] = int(a.size)
            if pred.object_threshold is not None: row["normal_native_object_alarm"] = int(s > pred.object_threshold)
        rows.append(row)
    points, samples, denominators = {}, {}, {}
    apids,normals = plan["AP_ids"],plan["normal_ids"]
    for key,column,index,draw in [
        ("mean_object_pixel_AP","object_pixel_AP",apids,plan["AP"]),
        *[(k,k,apids,plan["AP"]) for k in budget_keys],
        *[(k,k,normals,plan["normal"]) for k in ["normal_false_area","normal_map_any_alarm","normal_native_object_alarm"]]]:
        values = [rows[i][column] for i in index]
        # Missing threshold is intentional for raw PDN secondary, never discard selected objects.
        if any(v is None for v in values): points[key],samples[key],denominators[key] = None,None,0
        else: points[key],samples[key] = _mean_and_boot(values,draw); denominators[key] = len(values)
    object_metric = binary_metrics(bad,scores); object_boot = bootstrap_binary_metrics(bad,scores,plan["detection"])
    for key,m in [("object_AP","AP"),("object_AUROC","AUROC")]:
        points[key],samples[key],denominators[key] = object_metric[m],object_boot[m],n
    if pred.map_threshold is not None and len(normals):
        false_pixels = sum(rows[i]["normal_false_positions"] for i in normals)
        normal_pixels = sum(rows[i]["normal_total_positions"] for i in normals)
        points["normal_pooled_false_area"] = false_pixels/normal_pixels
        # Per-category N,V,H,W cache has equal positions/object, so pooled and
        # object-mean areas coincide. Resampling still uses complete objects.
        samples["normal_pooled_false_area"] = samples["normal_false_area"]
        denominators["normal_pooled_false_area"] = len(normals)
    else:
        points["normal_pooled_false_area"],samples["normal_pooled_false_area"],denominators["normal_pooled_false_area"] = None,None,0
    pooled = binary_metrics(gt[valid], maps[valid])
    points["pooled_pixel_AP"],points["pooled_pixel_AUROC"] = pooled["AP"],pooled["AUROC"]
    denominators["pooled_pixel_AP"] = denominators["pooled_pixel_AUROC"] = int(valid.sum())
    for k in ["pooled_pixel_AP","pooled_pixel_AUROC"]: samples[k] = None
    details = dict(objects=n, normal_objects=int((~bad).sum()), anomalous_objects=int(bad.sum()),
                   nominal_calibration_objects=pred.provenance.get("nominal_calibration_objects"),
                   actual_calibration_objects=pred.provenance.get("actual_calibration_objects"),
                   valid_pixel_GT_objects=int(valid.sum()), invalid_pixel_GT_objects=int((~valid).sum()),
                   valid_positive_anomalous_objects=int(plan["eligible"].sum()),
                   valid_empty_anomalous_objects=int((bad&valid&(positive==0)).sum()),
                   invalid_anomalous_objects=int((bad&~valid).sum()), valid_pixel_locations=int(gt[valid].size),
                   valid_positive_pixel_locations=pooled["positives"], real_views_per_object=int(gt.shape[1]),
                   standard_hw=list(config.expected_hw), AP_denominator_rule="valid positive anomalous objects only; null for empty/invalid",
                   object_denominator_rule="all labeled objects, including empty/invalid pixel GT",
                   budget_denominator_rule="same valid positive anomalous objects; floor budget across actual views",
                   false_alarm_denominator_rule="all heldout metadata-normal objects, fixed reviewed actual-subset calibration thresholds (nominal64); no refit",
                   pooled_pixel_rule="all pixels of valid pixel GT objects, including metadata-normal and valid zero-positive anomalies")
    return rows,points,samples,denominators,details


def evaluate_study(cache, predictions, output, config=StatisticsConfig(), *, synthetic_fixture=False, validated_inputs=None):
    'Compute statistics from explicitly validated arrays or synthetic fixtures.'
    require(synthetic_fixture or validated_inputs is not None, 'Real statistics require an explicit validated input/cache binding')
    out = Path(output); out.mkdir(parents=True,exist_ok=False)
    require(config.replicates == 2000 and config.seed == SEED, "Fixed resampling seed or replicate count changed")
    require(len(set(config.expected_systems)) == len(config.expected_systems), "Duplicate expected system")
    require(len(set(config.comparisons)) == len(config.comparisons) and config.primary_pair in config.comparisons,
            "Duplicate comparisons or absent primary pair")
    groups = {}
    for entry in cache.values():
        records = entry["records"]; require(bool(records), "Empty GT category")
        key = (records[0]["source"],records[0]["category"])
        require(key not in groups, "Repeated GT group")
        groups[key] = (entry["gt256"],records)
    expected = [(s,c) for s,cats in config.source_categories for c in cats]
    require(set(groups) == set(expected), "Fixed source/category scope incomplete or changed")
    bygroup = {k:{} for k in expected}
    for p in predictions:
        k = (p.source,p.category); require(k in bygroup and p.name not in bygroup[k], "Unknown/duplicate prediction system/group")
        bygroup[k][p.name] = p
    for k in expected: require(set(bygroup[k]) == set(config.expected_systems), "Fixed system set incomplete: "+str(k))
    protocol = dict(status="SYNTHETIC_FIXTURE" if synthetic_fixture else 'SEALED_COHORT_STATISTICS',
                    configuration=config.as_json(), validated_inputs=validated_inputs,
                    statistics_script_sha256=sha(__file__), ground_truth_access="explicit prepared sealed cache only; no archive/RGB/model/GPU interface")
    write(out/"STATISTICS_PROTOCOL.json",protocol)
    plans,exported = make_plans(groups,config); np.savez(out/"WHOLE_OBJECT_BOOTSTRAP_DRAWS.npz",**exported)
    estimates,objects,all_points,all_samples,counts,system_provenance = [],[],{}, {},{},[]
    for source,cat in expected:
        key = (source,cat); all_points[key],all_samples[key],counts[key] = {},{},{}
        gt,records = groups[key]
        for name in config.expected_systems:
            pred = bygroup[key][name]
            rows,points,samples,denominators,details = evaluate_one(pred,gt,records,plans[key],config)
            objects.extend(rows); all_points[key][name],all_samples[key][name],counts[key][name] = points,samples,details
            system_provenance.append(dict(source=source,category=cat,system=name,**pred.provenance))
            for endpoint,value in points.items():
                ci = interval(samples[endpoint])
                estimates.append(dict(scope="category",source=source,category=cat,system=name,endpoint=endpoint,value=value,
                                      denominator_objects=denominators[endpoint],CI95_lower=ci["lower"] if ci else None,
                                      CI95_upper=ci["upper"] if ci else None,
                                      interval_scope="pointwise95% whole-object" if ci else "point estimate only; undefined stays null"))
        print(json.dumps(dict(statistics_group=source+"::"+cat,systems=len(config.expected_systems),objects=len(records))),flush=True)
    contrasts = []
    endpoints = list(next(iter(next(iter(all_points.values())).values())))
    for source,cats in config.source_categories:
        for name in config.expected_systems:
            for endpoint in endpoints:
                values = [all_points[(source,c)][name][endpoint] for c in cats]
                bb = [all_samples[(source,c)][name][endpoint] for c in cats]
                value = float(np.mean(values)) if all(x is not None for x in values) else None
                bs = np.mean(np.stack(bb),0) if all(x is not None for x in bb) else None
                ci = interval(bs)
                estimates.append(dict(scope="fixed_categories_equal_weight_macro",source=source,category=None,system=name,
                                      endpoint=endpoint,value=value,denominator_categories=len(cats),
                                      missing_categories=";".join(c for c,v in zip(cats,values) if v is None),
                                      CI95_lower=ci["lower"] if ci else None,CI95_upper=ci["upper"] if ci else None,
                                      interval_scope="pointwise95% whole-object" if ci else "point estimate only; incomplete fixed macro stays null"))
        for left,right in config.comparisons:
            require(left in config.expected_systems and right in config.expected_systems,"Comparison system absent from fixed set")
            for endpoint in endpoints:
                primary = (left,right) == config.primary_pair and endpoint == "mean_object_pixel_AP"
                differences,boots = [],[]
                for cat in cats:
                    points,samples = all_points[(source,cat)],all_samples[(source,cat)]
                    a,b = points[left][endpoint],points[right][endpoint]
                    aa,bb = samples[left][endpoint],samples[right][endpoint]
                    delta = a-b if a is not None and b is not None else None
                    bs = aa-bb if aa is not None and bb is not None else None
                    ci = interval(bs); differences.append(delta); boots.append(bs)
                    contrasts.append(dict(scope="category",source=source,category=cat,endpoint=endpoint,left=left,right=right,
                        difference=delta,CI95_lower=ci["lower"] if ci else None,CI95_upper=ci["upper"] if ci else None,
                        CI975_lower=None,CI975_upper=None,primary_source_macro=False,
                        interpretation="pointwise95%, secondary category estimate" if ci else "point estimate only / undefined"))
                delta = float(np.mean(differences)) if all(d is not None for d in differences) else None
                bs = np.mean(np.stack(boots),0) if all(d is not None for d in boots) else None
                ci,bonf = interval(bs),interval(bs,.975) if primary else None
                contrasts.append(dict(scope="fixed_categories_equal_weight_macro",source=source,category=None,
                    endpoint=endpoint,left=left,right=right,difference=delta,
                    CI95_lower=ci["lower"] if ci else None,CI95_upper=ci["upper"] if ci else None,
                    CI975_lower=bonf["lower"] if bonf else None,CI975_upper=bonf["upper"] if bonf else None,
                    primary_source_macro=primary,
                    missing_categories=";".join(c for c,v in zip(cats,differences) if v is None),
                    interpretation="97.5% Bonferroni across two fixed primary source macros, with95% interval also shown" if primary else
                                   ("pointwise95% secondary, no familywise claim" if ci else "point estimate only / incomplete fixed macro")))
    csv_write(out/"PER_OBJECT_ENDPOINTS.csv",objects); csv_write(out/"ESTIMATES.csv",estimates); csv_write(out/"PAIRED_CONTRASTS.csv",contrasts)
    write(out/"COUNTS_AND_DENOMINATORS.json",[{"source":s,"category":c,"systems":counts[(s,c)]} for s,c in expected])
    write(out/"PREDICTION_PROVENANCE.json",system_provenance)
    primary = [r for r in contrasts if r["primary_source_macro"]]
    write(out/"RESULTS.json",dict(protocol=protocol, primary=primary, estimates=estimates, contrasts=contrasts,
                                  limitations="Fixed category, reference and calibration sets; only whole query objects resampled. No repeated deterministic inference CI, no new result imputation, no pixel/view CI. Pooled pixel metrics are point estimates. Raw/calibrated identities never selected by category."))
    write(out/"COMPLETE.json",dict(status="SYNTHETIC_COMPLETE" if synthetic_fixture else 'STATISTICS_COMPLETE',
        script_sha256=sha(__file__), results_sha256=sha(out/"RESULTS.json"), bootstrap_draws_sha256=sha(out/"WHOLE_OBJECT_BOOTSTRAP_DRAWS.npz"),
        GPU_used=False, new_RGB_read=False, fitting=False, original_predictions_modified=False))
    return read(out/"RESULTS.json")

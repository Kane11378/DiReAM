"""Posthoc AP contrasts from saved A176 whole-unit outputs.

No encoder, fitting, prediction, GT generation or calibration occurs here.
Input extraction and integrity evidence are in INPUT_PROVENANCE.json.
Run with --inputs INPUT_DIRECTORY --output NEW_DIRECTORY for a portable rerun.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

SEED = 10220261017
DRAWS = 2000
SOURCES = {"MANTA": ["maize", "paddy"], "VisA": ["candle", "pcb2"]}
PAIRS = [("M_coreset04", "predicted_teacher04"),
         ("within_I01", "within_N01_fast")]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def save_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.inputs / "INPUT_PROVENANCE.json").read_text(encoding="utf-8"))
    for pin in manifest["portable_inputs"]:
        assert sha(args.inputs / pin["file"]) == pin["sha256"], pin["file"]
    frozen_order = json.loads((args.inputs / "BOOTSTRAP_UNIT_ORDER.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    data, order, draws, counts = {}, {}, {}, []
    # Exactly the original script's global sorted category order. Generating a
    # new RNG per source/pair would silently change the historical shared draws.
    for cat in sorted(frozen_order):
        table = rows(args.inputs / f"PER_UNIT_{cat}.csv")
        methods = sorted({x["method"] for x in table})
        for method in methods:
            selected = [x for x in table if x["method"] == method]
            ids = [x["global_identity"] for x in selected]
            assert len(ids) == len(set(ids)), (cat, method, "duplicate unit")
            data[cat, method] = {x["global_identity"]: x for x in selected}
        all_ids = sorted(data[cat, methods[0]])
        eligible = [ident for ident in all_ids if data[cat, methods[0]][ident]["object_pixel_AP"] not in ("", None)]
        for method in methods:
            selected = data[cat, method]
            assert sorted(selected) == all_ids
            assert [ident for ident in all_ids if selected[ident]["object_pixel_AP"] not in ("", None)] == eligible
            assert all(selected[ident]["pixel_AP_eligible"] == "True" and selected[ident]["object_anomalous"] == "True" for ident in eligible)
        assert eligible == frozen_order[cat], (cat, "old ordered eligible IDs changed")
        order[cat] = eligible
        draws[cat] = rng.integers(0, len(eligible), size=(DRAWS, len(eligible)))
        counts.append(dict(category=cat, library="A176", queries_per_profile=len(all_ids), AP_eligible_units=len(eligible), real_views_per_unit=5 if cat in SOURCES["MANTA"] else 1))

    def estimate(cats, left, right):
        category_differences, sampled = [], []
        for cat in sorted(cats):
            difference = np.array([float(data[cat, left][ident]["object_pixel_AP"]) - float(data[cat, right][ident]["object_pixel_AP"]) for ident in order[cat]])
            category_differences.append(float(difference.mean()))
            sampled.append(difference[draws[cat]].mean(axis=1))
        low, high = np.quantile(np.mean(sampled, axis=0), [.025, .975])
        return float(np.mean(category_differences)), float(low), float(high)

    # A meaningful positive control: reproduce original stored intervals under
    # the same data, category ordering and shared draw stream to floating error.
    prior = rows(args.inputs / "ORIGINAL_PAIRED_CONTRASTS.csv")
    verification = []
    for source, cats in SOURCES.items():
        old = next(x for x in prior if x["source"] == source and x["library"] == "A176" and x["left"] == "within_M05" and x["right"] == "within_I01" and x["endpoint"] == "object_pixel_AP")
        actual = estimate(cats, old["left"], old["right"])
        expected = tuple(float(old[k]) for k in ["mean_difference", "pointwise95_lower", "pointwise95_upper"])
        error = max(abs(a-b) for a, b in zip(actual, expected))
        assert error < 1e-14, (source, actual, expected)
        verification.append(dict(source=source, left=old["left"], right=old["right"], maximum_absolute_reproduction_error=error))

    category, source_results = [], []
    for left, right in PAIRS:
        for source, cats in SOURCES.items():
            for cat in cats:
                left_values = np.array([float(data[cat, left][ident]["object_pixel_AP"]) for ident in order[cat]])
                right_values = np.array([float(data[cat, right][ident]["object_pixel_AP"]) for ident in order[cat]])
                category.append(dict(source=source, category=cat, library="A176", left=left, right=right, eligible_units=len(order[cat]), left_unit_AP=float(left_values.mean()), right_unit_AP=float(right_values.mean()), mean_difference=float((left_values-right_values).mean()), difference_percentage_points=float((left_values-right_values).mean()*100)))
            point, low, high = estimate(cats, left, right)
            source_results.append(dict(source=source, library="A176", left=left, right=right, endpoint="object_pixel_AP", categories=len(cats), eligible_units=sum(len(order[c]) for c in cats), mean_difference=point, pointwise95_lower=low, pointwise95_upper=high, difference_percentage_points=100*point, lower_percentage_points=100*low, upper_percentage_points=100*high, replicates=DRAWS, bootstrap_seed=SEED, analysis_status="posthoc requested reanalysis of already-seen saved outputs", scope="equal fixed-category means; paired complete eligible anomalous units within category; normal library/calibration fixed; no familywise/equivalence/transport guarantee"))
    save_csv(args.output / "PAIRED_AP_CATEGORY_POINTS.csv", category)
    save_csv(args.output / "PAIRED_AP_SOURCE_CONTRASTS.csv", source_results)
    save_csv(args.output / "UNIT_COUNTS.csv", counts)
    save_csv(args.output / "HISTORICAL_INTERVAL_REPRODUCTION.csv", verification)
    record = dict(status="COMPLETE_POSTHOC_DESCRIPTIVE_REANALYSIS", script_sha256=sha(Path(__file__)), input_manifest_sha256=sha(args.inputs / "INPUT_PROVENANCE.json"), bootstrap_seed=SEED, replicates=DRAWS, sorted_category_draw_order=sorted(frozen_order), source_rows=len(source_results), category_rows=len(category), query_or_model_runs=0, calibration_refits=0, maximum_historical_interval_reproduction_error=max(x["maximum_absolute_reproduction_error"] for x in verification), outputs=[dict(file=p.name, sha256=sha(p)) for p in sorted(args.output.glob("*.csv"))])
    (args.output / "COMPLETE.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=record["status"], source_results=source_results, historical_max_error=record["maximum_historical_interval_reproduction_error"]), indent=2))

if __name__ == "__main__":
    main()

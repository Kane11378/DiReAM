"""Check explicit datasets, weights, source files and normal-state resources.

Writes a JSON resource-check report without fitting a model, predicting images
or reading annotation pixels. Device and integration checks are separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(path):
    """Compare one Windows location consistently with or without Win32 prefix."""
    value = str(path)
    if os.name == "nt":
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
    return Path(value).resolve()


def package_member(root, name):
    require(isinstance(name, str) and name, "Unsafe package-relative path")
    relative = PurePosixPath(name)
    require(isinstance(name, str) and name and not relative.is_absolute()
            and ".." not in relative.parts and "\\" not in name and ":" not in name,
            "Unsafe package-relative path")
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path.is_file() and not (root / name).is_symlink(),
            "Missing or unsafe package member: " + name)
    return path


def verify_package(root):
    """Validate inventory before importing any packaged executable source."""
    inventory = read(root / "MANIFEST.json")
    names = set()
    for row in inventory["files"]:
        name = row["path"]
        require(name.casefold() not in names, "Duplicate package member: " + name)
        names.add(name.casefold())
        path = package_member(root, name)
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                "Package member changed: " + name)
    actual = {p.relative_to(root).as_posix().casefold() for p in root.rglob("*")
              if p.is_file() and p.relative_to(root).as_posix() != "MANIFEST.json"}
    require(names == actual, "Package inventory differs; extract into an empty directory")
    from source_identity import verify_sources
    identity = verify_sources(root)
    return dict(package_files=len(names), scientific_source_pins=identity["scientific_source_pins"],
                reviewed_current_derivations=identity["reviewed_current_derivations"],
                current_source_projection_sha256=identity["current_source_projection_sha256"],
                package_manifest_sha256=sha(root / "MANIFEST.json"),
                portable_source_pins_sha256=identity["portable_source_pins_sha256"])


def packaged_modules(root):
    sys.path.insert(0, str(root / "code"))
    values = {}
    for name in ('dataset_bindings', 'image_inference', 'inference_runner', 'matching_backend'):
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve() == (root / "code" / (name + ".py")).resolve(),
                "Another reader source is already imported: " + name)
        values[name] = module
    return values


def select_config(plan, category, profile, budget):
    matches = [c for c in plan["configs"] if c["category"] == category
               and c["profile_id"] == profile and c["budget_id"] == budget]
    require(len(matches) == 1, "Profile/category/budget combination is outside the fixed protocol")
    return dict(matches[0])


def selected_rows(group, mode):
    if mode == "all":
        return [(role, row) for role in ("reference", "calibration", "query") for row in group[role]]
    chosen = [(role, group[role][0]) for role in ("reference", "calibration", "query")]
    anomaly = next((row for row in group["query"] if row["label_stratum"] == "anomaly"), None)
    if anomaly is not None and anomaly["global_identity"] != chosen[-1][1]["global_identity"]:
        chosen.append(("query", anomaly))
    return chosen


def safe_external_member(source, name):
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and ".." not in rel.parts and "\\" not in name and ":" not in name,
            "Dataset member escapes explicit source")
    path = (source / name).resolve()
    require(path.is_relative_to(source.resolve()) and path.is_file()
            and not (source / name).is_symlink(), "Missing or unsafe RGB member: " + name)
    return path


def source_payloads(source, selected, archive_identity):
    """Directory or original archive; compressed archives need only one pass."""
    if source.is_dir():
        for role, row in selected:
            yield role, row, safe_external_member(source, row["rgb_member"]).read_bytes()
        return
    require(source.is_file(), "Bound image source is missing")
    require(source.stat().st_size == archive_identity["bytes"] and sha(source) == archive_identity["sha256"],
            "Original dataset archive bytes/SHA differ")
    if archive_identity["kind"] == "tar":
        with source.open("rb") as stream:
            for role, row in selected:
                stream.seek(row["rgb_offset_data_uncompressed"])
                yield role, row, stream.read(row["rgb_member_bytes"])
        return
    wanted = {row["rgb_member"]: (role, row) for role, row in selected}
    require(len(wanted) == len(selected), "Repeated selected RGB member")
    payloads = {}
    with tarfile.open(source, mode="r|gz") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            require(member.isfile() and not member.issym() and not member.islnk(), "RGB archive member is not regular")
            require(member.size == wanted[member.name][1]["rgb_member_bytes"], "RGB archive member size differs")
            payloads[member.name] = archive.extractfile(member).read()
            if len(payloads) == len(wanted):
                break
    require(set(payloads) == set(wanted), "Selected RGB members missing from archive")
    for role, row in selected:
        yield role, row, payloads[row["rgb_member"]]


def inspect_rgb(source, selected, archive_identity):
    import numpy as np
    from PIL import Image
    result = []
    for role, row, payload in source_payloads(source, selected, archive_identity):
        require(len(payload) == row["rgb_member_bytes"]
                and hashlib.sha256(payload).hexdigest() == row["source_payload_sha256"],
                "RGB member bytes/SHA differ: " + row["rgb_member"])
        with Image.open(io.BytesIO(payload)) as image:
            require(list(image.size) == row["composite_wh"], "Original image geometry changed")
            rgb = np.asarray(image.convert("RGB"))
        width, height = row["composite_wh"]
        views = []
        for x0, y0, x1, y1 in row["real_view_boxes"]:
            require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, "Invalid actual-view box")
            view = rgb[y0:y1, x0:x1].copy()
            require(view.dtype == np.uint8 and view.shape[-1] == 3, "RGB decode dtype/channels changed")
            views.append(dict(shape=list(view.shape), decoded_bytes_sha256=hashlib.sha256(view.tobytes()).hexdigest()))
        result.append(dict(role=role, global_identity=row["global_identity"],
                           rgb_member=row["rgb_member"], source_payload_sha256=row["source_payload_sha256"],
                           label_stratum=row["label_stratum"], actual_views=views))
    return result


def inspect_metrics(rows):
    import numpy as np
    result = []
    for row in rows:
        with np.load(row["model_path"], allow_pickle=False) as data:
            require(set(data.files) == {"center", "matrix"}, "Normal metric field names differ")
            require(data["center"].shape == (384,) and data["matrix"].shape == (384, 384),
                    "Normal metric geometry differs")
            require(all(data[name].dtype.kind == "f" and np.isfinite(data[name]).all()
                        for name in data.files), "Invalid normal metric values/dtype")
            result.append(dict(slot_id=row["slot_id"], sha256=row["model_sha256"],
                               shapes={name: list(data[name].shape) for name in data.files},
                               dtypes={name: str(data[name].dtype) for name in data.files}))
    return result


def runtime_probe(root):
    code = "import json,torch; print(json.dumps({'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'cuda_build':torch.version.cuda}))"
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    run = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True,
                         timeout=30, env=env)
    if run.returncode == 0:
        return dict(status="IMPORT_CHECKED", exit_code=0, **json.loads(run.stdout.strip().splitlines()[-1]))
    detail = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "Runtime import failed"
    # Store resource identities without account-specific paths.
    for path in sorted({str(root), str(sys.prefix), str(Path(sys.executable).parent)}, key=len, reverse=True):
        detail = detail.replace(path, "<LOCAL_RUNTIME_OR_READER>")
    return dict(status="IMPORT_FAILED", exit_code=run.returncode, diagnostic=detail,
                model_constructed=False, cuda_work_performed=False)


def inspect(args):
    root = canonical(args.reader)
    result = dict(schema="diream-public-resource-preflight-v1", checked_utc=datetime.now(timezone.utc).isoformat(),
                  producer_sha256=sha(__file__),
                  status="PREFLIGHT_INCOMPLETE", raw_inference_performed=False, runtime_report_produced=False,
                  model_constructed=False, prediction_performed=False, GT_reads=0)
    result["source_integrity"] = verify_package(root)
    modules = packaged_modules(root)
    portable, raw, runner, backend = [modules[x] for x in ('dataset_bindings', 'image_inference', 'inference_runner', 'matching_backend')]
    config = select_config(runner.plan(), args.category, args.profile, args.budget)
    result["configuration"] = {k: config[k] for k in ("category", "profile_id", "budget_id", "cohort_id", "config_id", "state_id", "k_per_slot")}
    group = portable.role_group(config)
    require(all(row["label_stratum"] == "normal" for role in ("reference", "calibration") for row in group[role]),
            "Fit or calibration contains an abnormal unit")
    result["roles"] = dict(status="ORDERED_ROLES_VALIDATED", counts={role: len(group[role]) for role in ("reference", "calibration", "query")},
                           membership_digest=group["membership_digest"], role_source_sha256=group["source_sha256"], cross_role_leakage=False)
    target, physical = portable.budget(config, portable.defaults())
    result["normal_metrics"] = inspect_metrics(physical)
    result["budget"] = dict(k_per_slot=target, clipped=False, source="fixed original physical budgets")
    resources, sources = raw.load_bindings(args.bindings)
    resource_errors = raw.check(resources, sources, args.category)
    require(not resource_errors, "Missing/changed explicit resources: " + json.dumps([{k:v for k,v in row.items() if k != "path"} for row in resource_errors]))
    result["bindings_sha256"] = sha(args.bindings)
    result["weight"] = dict(sha256=sha(resources["weights"]["anomalydino"]), model="DINOv2 ViT-S/14 no-register FP32")
    source_name = "MANTA" if args.category == "paddy" else "VisA"
    source = Path(sources[source_name])
    archive_identity = read(root / 'protocol/evaluation/original/EVALUATOR_SOURCES.json')["gt_sources"][args.category]
    selected = selected_rows(group, args.members)
    result["RGB"] = dict(selection=args.members, selection_rule="all ordered roles" if args.members == "all" else "first reference, first calibration, first query, first anomalous query when distinct",
                         checked_count=len(selected), total_role_count=sum(len(group[role]) for role in ("reference", "calibration", "query")),
                         source=source_name, source_kind="directory" if source.is_dir() else archive_identity["kind"],
                         archive_identity_checked=not source.is_dir(), records=inspect_rgb(source, selected, archive_identity))
    anomaly_rows = [row for row in group["query"] if row["label_stratum"] == "anomaly"]
    if source.is_dir():
        for row in anomaly_rows:
            annotation = safe_external_member(source, row["gt_member"])
            require(annotation.stat().st_size == row["gt_member_bytes"], "GT member byte count differs")
    result["annotations"] = dict(status="ARCHIVE_IDENTITY_CHECKED_PIXELS_UNOPENED" if source.is_file() else "MEMBER_EXISTENCE_SIZE_CHECKED_PIXELS_UNOPENED",
                                 anomalous_query_members=len(anomaly_rows),
                                 individual_GT_payload_checks="deferred to validated complete-prediction evaluation",
                                 GT_reads=0, evaluator='evaluation.evaluate_predictions; requires full valid predictions')
    result["state"] = dict(status="NOT_PROVIDED", producer="inference.py fit; CUDA and actual execution evidence required")
    if args.state_dir is not None:
        state = backend.validate_state_manifest(config, portable.defaults(), args.state_dir.resolve())
        result["state"] = dict(status="PRODUCTION_STATE_VALIDATED", state_identity_sha256=state["state_identity_sha256"],
                               state_manifest_sha256=sha(args.state_dir / "STATE.json"), source_kind=state["source_kind"])
    result["device"] = dict(requested=args.device, production_required="cuda", precision="FP32 encoder; FP64 mean/transform then FP32",
                            supported_for_production=args.device == "cuda", CPU_adaptation_performed=False)
    if args.probe_runtime:
        result["runtime"] = runtime_probe(root)
    result["historical_reuse"] = dict(requested=False, old_cache_or_prediction_payload_required=False,
                                      scope="New raw_rebuild resources only; historical parity was not measured")
    result['remaining_runtime_checks'] = ['Current CUDA resource report is not included in this resource-only check', 'Software, CUDA and integration reports are needed for the raw-rebuild bundle']
    if args.device == "cpu":
        result['remaining_runtime_checks'].insert(0, 'matching_backend._gpu rejects CPU: no scientific CPU fallback')
    if args.state_dir is None:
        result['remaining_runtime_checks'].append("No newly built and validated exact-budget production bank was supplied")
    if result.get("runtime", {}).get("status") == "IMPORT_FAILED":
        result['remaining_runtime_checks'].insert(0, "Torch import failed on this runtime")
    if args.members != "all":
        result['remaining_runtime_checks'].append("RGB byte/decode check covered selected sample members only")
    result["status"] = 'RESOURCE_CHECK_COMPLETE'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--category", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--budget", choices=("B", "half_B", "double_B", "full"), default="B")
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--members", choices=("sample", "all"), default="sample")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--probe-runtime", action="store_true", help="Import Torch in an isolated child; no model or CUDA work")
    parser.add_argument("--state-dir", type=Path, help="Optionally validate an existing exact production state; no construction")
    parser.add_argument("--output", type=Path, required=True, help='New JSON resource-check report outside the package; no overwrite')
    args = parser.parse_args(argv)
    root, target = canonical(args.reader), canonical(args.output)
    require(not target.is_relative_to(root), 'Resource-check output must be outside the package')
    require(not target.exists(), 'Resource-check output already exists')
    result = inspect(args)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", 'raw_inference_performed', "model_constructed", "prediction_performed", "GT_reads")}, indent=2))
    # Exit 0 reports completed resource checks; it does not run image inference.
    # Model construction, predictions and annotation reads are reported explicitly.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print("raw preflight: " + str(error), file=sys.stderr)
        raise SystemExit(2)

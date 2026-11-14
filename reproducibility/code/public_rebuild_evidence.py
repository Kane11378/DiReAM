"""Check a raw-image rebuild environment and record validation reports.

Resource, software, CUDA and small integration checks have distinct reports.
Reports bind the source, package and resource identities used by a rebuild.
A failed command cannot produce a successful completion report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_raw_preflight as preflight
from public_raw_preflight import read, require, sha

SCHEMA = "diream-current-raw-rebuild-evidence-v1"
PACKAGES = ("torch", "torchvision", "numpy", "scipy", "Pillow",
            "opencv-python-headless", "scikit-learn", "tqdm")


def now():
    return datetime.now(timezone.utc).isoformat()


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def new_output(root, output):
    root, output = preflight.canonical(root), preflight.canonical(output)
    require(not output.is_relative_to(root) and not output.exists(),
            "Choose a new output outside the reader")
    output.mkdir(parents=True)
    # Explicit Win32 long-path spelling keeps fixture scratch inside the caller's
    # chosen directory without a registry change, link or outside temporary root.
    if os.name == "nt" and not str(output).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(output))
    return output


def stamp(root, kind):
    verified = preflight.verify_package(root)
    return dict(schema=SCHEMA, kind=kind, checked_utc=now(),
                producer_sha256=sha(__file__),
                portable_source_pins_sha256=verified["portable_source_pins_sha256"],
                current_source_projection_sha256=verified["current_source_projection_sha256"],
                package_manifest_sha256=verified["package_manifest_sha256"],
                model_constructed=False, GT_reads=0)


def run_logged(arguments, output, name, timeout=60):
    environment = dict(os.environ)
    for key in ("TMP", "TEMP", "TMPDIR", "TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME"):
        environment[key] = str(output / "scratch")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    (output / "scratch").mkdir(exist_ok=True)
    run = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout, env=environment)
    stdout, stderr = output / (name + ".stdout.log"), output / (name + ".stderr.log")
    stdout.write_text(run.stdout, encoding="utf-8", newline="\n")
    stderr.write_text(run.stderr, encoding="utf-8", newline="\n")
    return dict(exit_code=run.returncode,
                stdout=dict(path=stdout.name, sha256=sha(stdout)),
                stderr=dict(path=stderr.name, sha256=sha(stderr))), run


def package_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "MISSING"
    return result


def machine(root, output):
    """No model; actual current CUDA/device ownership and an FP32 operation."""
    record = stamp(root, "machine")
    record.update(status='DEVICE_CHECK_INCOMPLETE', platform=platform.system(),
                  python=platform.python_version(), packages=package_versions())
    expected = read(root / "ENVIRONMENT.json")
    record["version_differences"] = {name: dict(expected=expected["packages"][name], actual=record["packages"][name])
                                     for name in PACKAGES if record["packages"][name] != expected["packages"][name]}
    probe_command = [sys.executable, "-I", "-B", "-c",
                     "import json,torch; print(json.dumps({'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'cuda_build':torch.version.cuda}))"]
    logged, run = run_logged(probe_command, output, "torch_import", 30)
    record["torch_import"] = logged
    if run.returncode:
        record["reason"] = "Torch import failed; see local command logs"
        new_json(output / "MACHINE.json", record)
        return record
    record["torch"] = json.loads(run.stdout.strip().splitlines()[-1])
    if record["platform"] != "Windows" or record["python"] != expected["python"] or record["version_differences"]:
        record["reason"] = "Selected raw-route environment differs from the recorded Windows/CUDA environment"
        new_json(output / "MACHINE.json", record)
        return record
    if not record["torch"]["cuda_available"]:
        record["reason"] = "CUDA unavailable; no scientific CPU fallback"
        new_json(output / "MACHINE.json", record)
        return record
    require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "Use explicit physical device 0 with no remapped CUDA_VISIBLE_DEVICES")
    try:
        logged_gpu, gpu = run_logged(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"], output, "gpu_inventory", 20)
        logged_jobs, jobs = run_logged(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"], output, "gpu_processes", 20)
        record.update(gpu_inventory=logged_gpu, gpu_processes=logged_jobs)
        require(gpu.returncode == 0 and jobs.returncode == 0, "Cannot establish actual GPU inventory/ownership")
        rows = [row.split(",", 2) for row in gpu.stdout.splitlines() if row.strip()]
        require(rows and rows[0][0].strip() == "0", "No physical GPU 0 in inventory")
        uuid = rows[0][1].strip()
        competitors = [line for line in jobs.stdout.splitlines() if line.strip() and uuid in line]
        require(not competitors, "Device 0 has running compute processes; resolve ownership before this check")
        record.update(device_index=0, device_uuid=uuid, gpu_compute_processes=competitors,
                      gpu_competition_resolved=True)
        cuda_code = "import json,torch; torch.cuda.set_device(0); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.use_deterministic_algorithms(True); x=torch.tensor([1.,2.],device='cuda',dtype=torch.float32); y=(x*x).sum(); torch.cuda.synchronize(); assert y.dtype==torch.float32 and float(y)==5.; print(json.dumps({'device_index':torch.cuda.current_device(),'result':float(y),'dtype':str(y.dtype),'tf32':False,'deterministic_algorithms':True}))"
        logged_cuda, cuda = run_logged([sys.executable, "-I", "-B", "-c", cuda_code], output, "cuda_fp32", 30)
        record["cuda_check"] = logged_cuda
        require(cuda.returncode == 0, "Actual CUDA FP32 operation failed")
        record["cuda_result"] = json.loads(cuda.stdout.strip().splitlines()[-1])
        record.update(status="CURRENT_CUDA_DEVICE_CHECKED", reason=None)
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        record["reason"] = str(error)
    new_json(output / "MACHINE.json", record)
    return record


def software(root, output, extra_site_packages=None):
    record = stamp(root, "software")
    record.update(status='SOFTWARE_CHECKS_INCOMPLETE', synthetic_only=True, commands=[])
    commands = [("core", [sys.executable, "-I", "-B", str(root / "code/run_reader_tests.py"), "--scratch", str(output / "scratch_core"), "--suite", "core"]),
                ("operators", [sys.executable, "-I", "-B", str(root / "code/run_reader_tests.py"), "--scratch", str(output / "scratch_operators"), "--suite", "operators"]),
                ("crossmodule", [sys.executable, "-I", "-B", str(root / 'tests/fixture_inference_integration.py'), "--output", str(output / "fixture_crossmodule")])]
    if extra_site_packages is not None:
        require(extra_site_packages.is_dir(), "Explicit software dependency directory missing")
        # Keep the selected CPU NumPy/Torch. Additional installed operator-only
        # modules are read from the explicit directory, without an installation.
        name, original = commands[1]
        script, arguments = original[3], original[4:]
        code = ("import sys,runpy,numpy,torch; "
                + "sys.path.append(" + repr(str(extra_site_packages.resolve())) + "); "
                + "sys.argv=" + repr([script] + arguments) + "; runpy.run_path(sys.argv[0],run_name='__main__')")
        commands[1] = (name, [sys.executable, "-I", "-B", "-c", code])
    for name, command in commands:
        logged, run = run_logged(command, output, name, 60)
        logged["name"] = name
        if name in ("core", "operators") and run.returncode == 0:
            logged["summary"] = json.loads(run.stdout.strip().splitlines()[-1])
            summary = logged["summary"]
            logged["skips_detected"] = bool(re.search(r"skipped(?:=|\s|'|\")", run.stderr, re.IGNORECASE))
            require(summary["tests"] > 0 and summary["failures"] == summary["errors"] == 0
                    and summary["synthetic_only"] is True and not logged["skips_detected"],
                    "Software suite did not execute cleanly without skips")
        record["commands"].append(logged)
        if run.returncode:
            record["reason"] = "Actual software command failed: " + name
            new_json(output / "SOFTWARE.json", record)
            return record
    fixture = output / "fixture_crossmodule/FIXTURE_RESULT.json"
    if not fixture.is_file():
        fixture = output / "fixture_crossmodule/FIXTURE_REPORT.json"
    # The CLI's own exit status verifies its synthetic numerical assertions.
    record.update(status="CURRENT_SOFTWARE_CHECKS_COMPLETE", reason=None)
    record["output_files"] = [{"path": p.relative_to(output).as_posix(), "sha256": sha(p)}
                              for p in sorted((output / "fixture_crossmodule").rglob("*")) if p.is_file()]
    new_json(output / "SOFTWARE.json", record)
    return record


def evidence(path, expected_sha, root, kind):
    path = Path(path).resolve()
    require(path.is_file() and sha(path) == expected_sha, "Current evidence bytes changed: " + kind)
    doc = read(path)
    require(doc.get("schema") == SCHEMA and doc.get("kind") == kind, "Wrong current evidence type: " + kind)
    require(doc.get("producer_sha256") == sha(__file__), "Evidence producer source differs")
    require(doc.get("portable_source_pins_sha256") == sha(root / 'protocol/evaluation/SOURCE_PINS.json'), "Evidence binds another portable source snapshot")
    require(doc.get("package_manifest_sha256") == sha(root / "MANIFEST.json"), "Evidence binds another package inventory")
    from source_identity import projection_sha256
    require(doc.get("current_source_projection_sha256") == projection_sha256(root), "Evidence binds another current source projection")
    for command in doc.get("commands", []):
        for name in ("stdout", "stderr"):
            pin = command[name]
            target = preflight.package_member(path.parent, pin["path"])
            require(sha(target) == pin["sha256"], "Software command log changed")
        require(command["exit_code"] == 0, "A software command failed")
    if kind == "machine":
        for name in ("torch_import", "gpu_inventory", "gpu_processes", "cuda_check"):
            command = doc.get(name)
            if command is None:
                continue
            require(command["exit_code"] == 0, "A machine command failed")
            for stream in ("stdout", "stderr"):
                pin = command[stream]
                target = preflight.package_member(path.parent, pin["path"])
                require(sha(target) == pin["sha256"], "Machine command log changed")
    for pin in doc.get("output_files", []):
        target = preflight.package_member(path.parent, pin["path"])
        require(sha(target) == pin["sha256"], "Evidence output changed")
    return doc


def check_components(root, resources_path, software_path, machine_path, *, config_id=None):
    resources = read(resources_path)
    require(resources.get("schema") == "diream-public-resource-preflight-v1"
            and resources.get("status") == 'RESOURCE_CHECK_COMPLETE', "Missing actual current resource preflight")
    require(resources.get("producer_sha256") == sha(preflight.__file__), "Resource preflight producer changed")
    from source_identity import projection_sha256
    require(resources["source_integrity"].get("current_source_projection_sha256") == projection_sha256(root), "Resource current source projection differs")
    require(resources["source_integrity"]["portable_source_pins_sha256"] == sha(root / 'protocol/evaluation/SOURCE_PINS.json')
            and resources["source_integrity"]["package_manifest_sha256"] == sha(root / "MANIFEST.json"), "Resource source snapshot differs")
    require(resources["RGB"]["selection"] == "all"
            and resources["RGB"]["checked_count"] == resources["RGB"]["total_role_count"], "Current production requires every selected role RGB member")
    require(resources["roles"]["cross_role_leakage"] is False and resources["budget"]["clipped"] is False, "Invalid roles/budget receipt")
    modules = preflight.packaged_modules(root)
    config = resources["configuration"]
    actual = preflight.select_config(modules['inference_runner'].plan(), config["category"], config["profile_id"], config["budget_id"])
    require(all(config[key] == actual[key] for key in config), "Resource receipt changed the fixed configuration")
    group = modules['dataset_bindings'].role_group(actual)
    require(resources["roles"]["membership_digest"] == group["membership_digest"]
            and resources["roles"]["role_source_sha256"] == group["source_sha256"]
            and resources["roles"]["counts"] == {role:len(group[role]) for role in ("reference", "calibration", "query")},
            "Resource receipt role membership/order differs")
    expected = [(role,row) for role in ("reference", "calibration", "query") for row in group[role]]
    observed = resources["RGB"]["records"]
    require(len(observed) == len(expected), "Resource receipt omits RGB records")
    for record, (role,row) in zip(observed, expected):
        require(record["role"] == role and record["global_identity"] == row["global_identity"]
                and record["rgb_member"] == row["rgb_member"]
                and record["source_payload_sha256"] == row["source_payload_sha256"], "Resource RGB identity/order differs")
    target, physical = modules['dataset_bindings'].budget(actual, modules['dataset_bindings'].defaults())
    require(resources["budget"]["k_per_slot"] == target and resources["normal_metrics"] == preflight.inspect_metrics(physical), "Resource metric/K receipt differs")
    require(resources["weight"]["sha256"] == modules['matching_backend'].WEIGHT_SHA and resources["GT_reads"] == 0, "Resource weight/GT scope differs")
    if config_id is not None:
        require(resources["configuration"]["config_id"] == config_id, "Evidence belongs to another exact configuration")
    sw = evidence(software_path, sha(software_path), root, "software")
    require(sw.get("status") == "CURRENT_SOFTWARE_CHECKS_COMPLETE" and [row["name"] for row in sw["commands"]] == ["core", "operators", "crossmodule"], "Current software checks are incomplete")
    for row in sw["commands"][:2]:
        summary = row["summary"]
        require(summary["tests"] > 0 and summary["errors"] == summary["failures"] == 0
                and row["skips_detected"] is False, "Software results invalid")
    mc = evidence(machine_path, sha(machine_path), root, "machine")
    require(mc.get("status") == "CURRENT_CUDA_DEVICE_CHECKED" and mc.get("gpu_competition_resolved") is True
            and mc.get("gpu_compute_processes") == [] and mc.get("version_differences") == {}, "Current CUDA/device checks are incomplete")
    expected_environment = read(root / "ENVIRONMENT.json")
    require(mc.get("packages") == package_versions()
            and mc.get("packages") == {name:expected_environment["packages"][name] for name in PACKAGES}
            and platform.system() == mc.get("platform") == "Windows"
            and platform.python_version() == mc.get("python") == expected_environment["python"], "Machine evidence belongs to another environment")
    require(all(name in mc for name in ("torch_import", "gpu_inventory", "gpu_processes", "cuda_check")), "Actual machine command receipts are missing")
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(mc["checked_utc"])).total_seconds()
    require(0 <= age <= 900, "Current CUDA ownership check is stale; regenerate the machine receipt and bundle")
    require(mc["cuda_result"] == dict(device_index=0, result=5.0, dtype="torch.float32", tf32=False, deterministic_algorithms=True), "Actual FP32 CUDA check differs")
    return resources, sw, mc


def integration(root, output, resources_path, software_path, machine_path, bindings):
    resources, _, _ = check_components(root, resources_path, software_path, machine_path)
    require(resources["configuration"]["category"] == "chewinggum", 'Prespecified CUDA integration fixture is chewinggum only')
    require(resources["bindings_sha256"] == sha(bindings), 'Integration resource bindings changed')
    modules = preflight.packaged_modules(root)
    portable, raw, runner, backend = [modules[x] for x in ('dataset_bindings', 'image_inference', 'inference_runner', 'matching_backend')]
    config = preflight.select_config(runner.plan(), **{k:resources["configuration"][v] for k,v in (("category","category"),("profile","profile_id"),("budget","budget_id"))})
    bound_resources, sources = raw.load_bindings(bindings)
    resource_file = output / "EXPLICIT_RESOURCES.json"
    new_json(resource_file, bound_resources)
    context = portable.defaults(dict(run_root=str(output), image_sources=sources, resources_json=str(resource_file), gpu_safety_checked=True))
    # The integration check uses the fixed small normal-only fixture.
    # It does not create full-study predictions or fitted-state outputs.
    def integration_runtime(context, **kwargs):
        require(not kwargs.get("timing"), 'The integration report does not cover request timing')
        check_components(root, resources_path, software_path, machine_path)
        context["gpu_safety_checked"] = True
        return dict(status='CUDA_INTEGRATION_CHECK')
    runner.validate_runtime = integration_runtime
    with runner.job_lock(output / "artifacts/GPU_SERIAL"):
        report = backend.integration_fixture(config, context, output / "fixture")
    require(report["status"] == "PASSED" and report["scope"] == 'CUDA_NORMAL_INTEGRATION_FIXTURE', 'Integration fixture did not complete')
    record = stamp(root, 'integration')
    record.update(status='CUDA_INTEGRATION_COMPLETE', software_example=True, production_state_created=False,
                  example_profile="four_fit_three_calibration_exact_K64_integration", report=report,
                  source_bindings_sha256=sha(bindings), model_constructed=True)
    record["output_files"] = [{"path": p.relative_to(output).as_posix(), "sha256": sha(p)}
                              for p in sorted((output / "fixture").rglob("*")) if p.is_file()]
    new_json(output / 'INTEGRATION.json', record)
    return record


def validate_bundle(root, path, *, config_id=None):
    root, path = Path(root).resolve(), Path(path).resolve()
    preflight.verify_package(root)
    bundle = read(path)
    require(bundle.get("schema") == SCHEMA and bundle.get("mode") == "raw_rebuild", "Require current raw-rebuild evidence schema")
    require(bundle.get('historical_reuse_included') is False and bundle.get('timing_included') is False, 'This report schema does not cover historical reuse or request timing')
    require(bundle.get("producer_sha256") == sha(__file__), "Bundle producer source changed")
    locations = {}
    for kind in ("resources", "software", "machine", 'integration'):
        pin = bundle["evidence"][kind]
        target = Path(pin["path"])
        target = target.resolve() if target.is_absolute() else (path.parent / target).resolve()
        require(sha(target) == pin["sha256"], "Bundle evidence bytes changed: " + kind)
        locations[kind] = target
    resources, _, _ = check_components(root, locations["resources"], locations["software"], locations["machine"], config_id=config_id)
    eg = evidence(locations['integration'], bundle["evidence"]['integration']["sha256"], root, 'integration')
    require(eg.get("status") == 'CUDA_INTEGRATION_COMPLETE' and eg.get("software_example") is True
            and eg.get("production_state_created") is False, 'Actual current CUDA integration evidence missing')
    report = eg["report"]
    require(report["scope"] == 'CUDA_NORMAL_INTEGRATION_FIXTURE'
            and report["formal_states_created"] is False and report["GT_reads"] == 0
            and len(report["fit_role"]) == 4 and len(report["calibration_role"]) == 3, 'Integration scope differs')
    require(len(report["rows"]) == 6 and all(row["status"] == "PASSED" and row["retained_rows"] == 64 for row in report["rows"]), 'Integration output checks incomplete')
    modules = preflight.packaged_modules(root)
    group = modules['dataset_bindings'].role_group(dict(category="chewinggum", profile_id="M_W_U", budget_id="B"))
    require(report["fit_role"] == [row["global_identity"] for row in group["reference"][:4]]
            and report["calibration_role"] == [row["global_identity"] for row in group["calibration"][:3]],
            'Integration fixed member identities differ')
    require(report["atol"] == 1e-6 and report["rtol"] == 1e-5
            and [row["profile_id"] for row in report["rows"]] == ["NativeNoRot_exact"] * 3 + ["I_W_U"] * 3,
            'Integration profile/tolerance differs')
    files = {row["path"] for row in eg["output_files"]}
    needed = {"fixture/FIXTURE_RESULT.json"} | {f"fixture/{kind}_normal_{index}.npz" for kind in ("native", "I") for index in range(3)}
    require(needed <= files, 'Integration prediction/report payloads are missing')
    require(read(locations['integration'].parent / "fixture/FIXTURE_RESULT.json") == report,
            'Integration report differs from its sealed fixture output')
    require(eg["source_bindings_sha256"] == resources["bindings_sha256"], 'Integration resource bindings differ')
    require(bundle["configuration"] == resources["configuration"], "Bundle exact configuration differs")
    return bundle


def check_raw_output_origin(context, configuration):
    """Keep old reuse artifacts out of current rebuild's existing-output paths.

    Original production validators still check every payload. This separate
    route boundary rejects historical origins before an existing-output fast
    path can accept a prediction or cached fitted state.
    """
    run = preflight.canonical(Path(context["run_root"]))
    for name in ("config_id", "state_id"):
        require(bool(re.fullmatch(r"[0-9a-f]{64}", configuration[name])), "Invalid exact output identity")
    prediction = run / "artifacts/p" / configuration["config_id"][:16]
    state = run / "artifacts/s" / configuration["state_id"][:16]
    for folder in (prediction, state):
        require(folder.resolve().is_relative_to(run), "Existing raw output escapes run root")
    candidates = run / "artifacts/nc"
    require(candidates.resolve().is_relative_to(run), "Existing native candidates escape run root")
    for candidate in candidates.glob("*/NATIVE_CANDIDATES.json"):
        require(candidate.resolve().is_relative_to(run), "Existing native candidate manifest escapes run root")
        require(not read(candidate).get("source_reuse"), "Current rebuild rejects historical native candidate pool")
    complete = prediction / "COMPLETE.json"
    if complete.exists():
        require(read(complete).get("source_kind") == "RAW_RECOMPUTED", "Current rebuild rejects historical prediction completion")
    progress = prediction / "PROGRESS.json"
    if progress.exists():
        doc = read(progress)
        require(doc.get("source_kind") in (None, "RAW_RECOMPUTED") and "source_descriptor_digest" not in doc,
                "Current rebuild rejects historical prediction progress")
    manifest = state / "STATE.json"
    if manifest.exists():
        doc = read(manifest)
        require("public_projection" not in doc and doc.get("source_kind") == "NEW_STATE_BINDING", "Current rebuild rejects historical projected state metadata")
        if doc.get("profile_family") == "diream":
            require(all(row.get("source_kind") == "RAW_RGB_REENCODED_FIT" and row.get("cache") is None for row in doc["slots"]),
                    "Current rebuild rejects historical or cached transformed state")
        else:
            candidate = preflight.canonical(Path(doc["native_state"]["candidate_manifest"]))
            require(candidate.is_relative_to(run), "Current rebuild requires direct-RGB candidates inside this run")
            require(not read(candidate).get("source_reuse"), "Current rebuild rejects historical native candidate slices")


def validate_runtime(context, formal=True, timing=False):
    """Explicit opt-in hook for raw rebuild; original state/prediction checks remain."""
    require(not timing, 'The raw-rebuild report does not cover request timing')
    for name in ("component_cache_mode", "native_candidate_mode", "state_reuse_mode"):
        require(context.get(name) == "raw_rebuild", 'Raw rebuilding requires original images rather than historical caches or states')
    require(not context.get("regression_baselines") and not context.get("historical_state_roots"), "Old reuse requires the historical route and successful original regressions")
    root = Path(context["reader_root"]).resolve()
    path = context.get("rebuild_evidence")
    require(path and context.get("rebuild_config_id"), "Supply current rebuild evidence and exact configuration identity")
    bundle = validate_bundle(root, path, config_id=context["rebuild_config_id"])
    check_raw_output_origin(context, bundle["configuration"])
    require(sha(context["bindings_path"]) == bundle["source_bindings_sha256"], "Current resource bindings changed")
    context["gpu_safety_checked"] = True
    original = root / 'protocol/evaluation/original'
    # Bind the checked rebuild to the fixed scientific protocol.
    # Current resources and environment were verified above.
    return dict(status='PROTOCOL_VERIFIED', current_evidence_schema=SCHEMA,
                current_rebuild_evidence_sha256=sha(path), historical_parity="NOT_MEASURED_NOT_REQUIRED_FOR_RAW_REBUILD",
                freeze_path=str(original / "PROTOCOL_FREEZE.json"), protocol_path=str(original / "PROTOCOL_RESOLVED.json"),
                protocol_sha256=sha(original / "PROTOCOL_RESOLVED.json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("machine", "software", 'integration', "assemble", "validate"))
    parser.add_argument("--reader", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--software", type=Path)
    parser.add_argument("--machine", type=Path)
    parser.add_argument('--integration', type=Path)
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--software-site-packages", type=Path, help="Optional explicit installed operator dependencies; CPU Torch/NumPy remain selected")
    args = parser.parse_args(argv)
    root = preflight.canonical(args.reader)
    if args.command == "validate":
        require(args.bundle is not None, "Supply --bundle")
        doc = validate_bundle(root, args.bundle)
        print(json.dumps(dict(status="CURRENT_RAW_REBUILD_EVIDENCE_VALID", configuration=doc["configuration"]), indent=2))
        return 0
    require(args.output is not None, "Supply a new --output directory")
    output = new_output(root, args.output)
    if args.command == "machine":
        doc = machine(root, output)
    elif args.command == "software":
        doc = software(root, output, args.software_site_packages)
    else:
        require(all((args.resources, args.software, args.machine)), "Supply actual --resources, --software and --machine receipts")
        if args.command == 'integration':
            require(args.bindings, "Supply --bindings")
            doc = integration(root, output, args.resources, args.software, args.machine, args.bindings)
        else:
            require(args.integration, 'Supply actual --integration receipt')
            resources, _, _ = check_components(root, args.resources, args.software, args.machine)
            doc = dict(schema=SCHEMA, mode="raw_rebuild", checked_utc=now(), producer_sha256=sha(__file__),
                       configuration=resources["configuration"], source_bindings_sha256=resources["bindings_sha256"],
                       historical_reuse_included=False, timing_included=False,
                       evidence={name: dict(path=os.path.relpath(str(path.resolve()).removeprefix("\\\\?\\"), str(output).removeprefix("\\\\?\\")), sha256=sha(path))
                                 for name, path in (("resources", args.resources), ("software", args.software), ("machine", args.machine), ('integration', args.integration))})
            new_json(output / "REBUILD_EVIDENCE.json", doc)
            validate_bundle(root, output / "REBUILD_EVIDENCE.json")
            doc["status"] = "CURRENT_RAW_REBUILD_EVIDENCE_VALID"
    print(json.dumps({key: doc[key] for key in ("status", "kind", "model_constructed") if key in doc}, indent=2))
    successful_statuses={'CURRENT_CUDA_DEVICE_CHECKED','CURRENT_SOFTWARE_CHECKS_COMPLETE','CUDA_INTEGRATION_COMPLETE','CURRENT_RAW_REBUILD_EVIDENCE_VALID'}
    return 0 if doc.get('status') in successful_statuses else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print("current rebuild evidence: " + str(error), file=sys.stderr)
        raise SystemExit(2)

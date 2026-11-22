"""Kaggle entrypoint for the CORD QLoRA JSON extraction benchmark (GPU T4).

Installs ``qlora_extraction`` from the pinned source dataset, loads the
CORD-v2 splits over HTTPS (ground_truth column only), freezes the analysis
plan, runs every job spec in a fresh subprocess for resource isolation,
then collects metrics and the ordering manifest. ``QLORA_SMOKE=1`` shrinks
data and epochs for a diagnostic run whose outputs are never committed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
BENCH = WORK / "benchmark"
SRC_SLUG = "qlora-json-extraction-src"
SRC_OWNER = "sreeharshak101"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _locate_src_dataset() -> Path:
    """Locate the mounted source dataset; explicit paths only, no fallback."""
    candidates = [
        KAGGLE_INPUT / "datasets" / SRC_OWNER / SRC_SLUG,
        KAGGLE_INPUT / SRC_SLUG,
    ]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists():
            return candidate
    found = sorted(
        p
        for p in KAGGLE_INPUT.rglob("pyproject.toml")
        if (p.parent / "src" / "qlora_extraction").is_dir()
    )
    if found:
        return found[0].parent
    listing = sorted(
        str(p) for p in KAGGLE_INPUT.iterdir()
    )
    raise FileNotFoundError(
        f"source dataset {SRC_OWNER}/{SRC_SLUG} not found under {KAGGLE_INPUT}; "
        f"top-level entries: {listing}"
    )


def _install_package(src: Path) -> Path:
    proj = WORK / "project"
    shutil.copytree(src, proj, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    _run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", str(proj)])
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "transformers>=4.46,<5",
            "peft>=0.13",
            "bitsandbytes>=0.43",
            "datasets>=3.2",
            "accelerate>=0.34",
            "pyarrow>=15",
            "fsspec[http]>=2024.6",
            "zss>=1.2",
            "pydantic>=2.8",
            "numpy>=2.0",
            "typer>=0.12",
        ]
    )
    return proj


def _git_sha(proj: Path) -> str:
    info = proj / "run_info.json"
    if info.exists():
        return str(json.loads(info.read_text()).get("git_sha", "unknown"))
    return "unknown"


def _gpu_snapshot() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi unavailable: {exc}"


def main() -> None:
    src = _locate_src_dataset()
    print(f"src dataset: {src}", flush=True)
    proj = _install_package(src)
    sha = _git_sha(proj)
    print(f"git_sha={sha}", flush=True)

    import torch  # noqa: PLC0415

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

    from transformers import AutoTokenizer  # noqa: PLC0415

    from qlora_extraction.data.cord import load_cord_split  # noqa: PLC0415
    from qlora_extraction.experiments import (  # noqa: PLC0415
        BenchmarkConfig,
        collect,
        job_specs,
        prepare,
    )
    from qlora_extraction.training import (  # noqa: PLC0415
        BASE_MODEL,
        BASE_MODEL_REVISION,
    )

    smoke = bool(os.environ.get("QLORA_SMOKE"))
    cfg = BenchmarkConfig(smoke=smoke)
    BENCH.mkdir(parents=True, exist_ok=True)

    t_load = time.monotonic()
    train_docs = load_cord_split("train")
    val_docs = load_cord_split("validation")
    test_docs = load_cord_split("test")
    print(
        f"loaded splits: train={len(train_docs)} val={len(val_docs)} test={len(test_docs)} "
        f"in {time.monotonic() - t_load:.1f}s",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_MODEL_REVISION
    )

    t_prepare = time.monotonic()
    plan = prepare(BENCH, train_docs, val_docs, test_docs, tokenizer, cfg)
    prepare_s = time.monotonic() - t_prepare
    print(f"prepare done in {prepare_s:.1f}s", flush=True)

    specs_dir = BENCH / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    # Reduce allocator fragmentation in the worker subprocesses; the QLoRA
    # smoke OOM showed ~1 GiB reserved-but-unallocated at the failure point.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    job_timings: dict[str, float] = {}
    for spec in job_specs(cfg, plan):
        spec_path = specs_dir / f"{spec['name']}.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        print(f"=== job {spec['name']} ({_gpu_snapshot()}) ===", flush=True)
        t0 = time.monotonic()
        _run(
            [
                sys.executable,
                "-m",
                "qlora_extraction.experiments.worker",
                "--spec",
                str(spec_path),
                "--work-dir",
                str(BENCH),
            ]
        )
        elapsed = time.monotonic() - t0
        job_timings[spec["name"]] = elapsed
        print(
            f"=== job {spec['name']} done in {elapsed:.1f}s ({_gpu_snapshot()}) ===",
            flush=True,
        )

    t_collect = time.monotonic()
    result = collect(BENCH, test_docs, val_docs, cfg)
    collect_s = time.monotonic() - t_collect
    print(f"collect done in {collect_s:.1f}s", flush=True)

    run_info = {
        "git_sha": sha,
        "smoke": smoke,
        "data_load_s": t_prepare - t_load,
        "prepare_s": prepare_s,
        "job_timings_s": job_timings,
        "collect_s": collect_s,
    }
    (BENCH / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    manifest = result["manifest"]
    print("=== MANIFEST ===", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    metrics = result["metrics"]
    test_metrics = metrics.get("test", {})
    for cond, block in test_metrics.items():
        print(
            f"{cond}: field_f1={block.get('field_f1'):.4f} "
            f"parse={block.get('parse_rate'):.3f} "
            f"schema={block.get('schema_valid_rate'):.3f}",
            flush=True,
        )
    if smoke:
        print("=== SMOKE TIMINGS ===", flush=True)
        for name, elapsed in job_timings.items():
            print(f"{name}: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()

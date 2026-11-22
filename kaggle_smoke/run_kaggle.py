"""Smoke kernel: reduced-work diagnostic run of BOTH benchmark parts.

Runs the main kernel script twice with ``QLORA_SMOKE=1``: part A into
``benchmark_a`` and part B into ``benchmark_b``, each doing prepare, its
job subset, and collect. Then ``scripts/merge_results.py`` merges the two
outputs, proving the split protocol end to end on tiny subsets. Smoke
outputs are diagnostics only and are never committed as results.
"""

import os
import runpy
import subprocess
import sys
from pathlib import Path

os.environ["QLORA_SMOKE"] = "1"

candidates = [
    p
    for p in Path("/kaggle/input").rglob("run_kaggle.py")
    if p.parent.name == "kaggle"
]
if not candidates:
    raise SystemExit("main kernel script not found in the source dataset")
target = sorted(candidates)[0]
print(f"smoke: delegating to {target}", flush=True)

work = Path("/kaggle/working")
for part, bench in (("A", "benchmark_a"), ("B", "benchmark_b")):
    os.environ["QLORA_PART"] = part
    os.environ["QLORA_BENCH_DIR"] = bench
    print(f"=== smoke part {part} -> {bench} ===", flush=True)
    sys.argv = [str(target)]
    runpy.run_path(str(target), run_name="__main__")

merge_script = work / "project" / "scripts" / "merge_results.py"
print(f"=== smoke merge: {merge_script} ===", flush=True)
subprocess.run(
    [
        sys.executable,
        str(merge_script),
        "--a",
        str(work / "benchmark_a"),
        "--b",
        str(work / "benchmark_b"),
        "--out",
        str(work / "benchmark_merged"),
    ],
    check=True,
)
print("=== smoke merge OK ===", flush=True)

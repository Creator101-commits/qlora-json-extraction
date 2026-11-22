"""Smoke kernel: reduced-work diagnostic run of the QLoRA benchmark.

Reuses the archived main kernel script from the source dataset with
``QLORA_SMOKE=1`` so the full pipeline (data load, schema inventory, dedup
audit, frozen plan, prompted jobs, one QLoRA seed at one epoch, collect)
is exercised cheaply before the full run. Smoke outputs are diagnostics
only and are never committed as results.
"""

import os
import runpy
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
sys.argv = [str(target)]
runpy.run_path(str(target), run_name="__main__")

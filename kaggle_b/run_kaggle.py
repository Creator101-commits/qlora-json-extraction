"""Part B kernel: seeds 1-2 and the 400-doc budget point.

Delegates to the main kernel script with ``QLORA_PART=B`` so both parts
share one frozen plan (proved by ``plan_sha256``) and disjoint job sets.
Launched by the lead after the weekly GPU quota reset.
"""

import os
import runpy
import sys
from pathlib import Path

os.environ["QLORA_PART"] = "B"

candidates = [
    p
    for p in Path("/kaggle/input").rglob("run_kaggle.py")
    if p.parent.name == "kaggle"
]
if not candidates:
    raise SystemExit("main kernel script not found in the source dataset")
target = sorted(candidates)[0]
print(f"part B: delegating to {target}", flush=True)
sys.argv = [str(target)]
runpy.run_path(str(target), run_name="__main__")

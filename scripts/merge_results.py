#!/usr/bin/env python3
"""Merge part A and part B kernel outputs into one benchmark result.

Usage: python scripts/merge_results.py --a DIR_A --b DIR_B --out DIR_OUT
Pure JSON; asserts both parts share git_sha and plan_sha256, have
disjoint job sets, and ordering_ok in both manifests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qlora_extraction.experiments.merge import merge_parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, type=Path, help="part A output dir")
    parser.add_argument("--b", required=True, type=Path, help="part B output dir")
    parser.add_argument("--out", required=True, type=Path, help="merged output dir")
    args = parser.parse_args()
    result = merge_parts(args.a, args.b, args.out)
    manifest = result["manifest"]
    print(f"merged git_sha={manifest['git_sha']} plan_sha256={manifest['plan_sha256'][:16]}...")
    print(f"conditions: {', '.join(manifest['conditions_run'])}")
    seeds = result["metrics"].get("qlora_seeds", {})
    print(f"qlora field_f1 mean={seeds.get('field_f1_mean')} sd={seeds.get('field_f1_sd')}")


if __name__ == "__main__":
    main()

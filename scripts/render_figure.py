"""Reproduce image.png from actual code outputs; no invented benchmark values.

Requires the project dependencies and an environment with Matplotlib installed.
Run from the repository root with PYTHONPATH=src python scripts/render_figure.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from qlora_extraction.data.cord import load_cord_dataset
from qlora_extraction.evaluation.runner import evaluate_extraction_condition
from qlora_extraction.models.lora import LoRALinear

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)
COLORS = ["#4169e1", "#10a68b", "#d95f45", "#8659cf", "#64748b"]
ROOT = Path(__file__).resolve().parents[1]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")


docs = [d for d in load_cord_dataset() if d.split == "test"]
conditions = ["heuristic_minimal", "heuristic_totals", "heuristic_items"]
metrics = [evaluate_extraction_condition(c, docs) for c in conditions]
x = np.arange(3)
axes[0].bar(
    x - 0.18, [m.schema_validity_rate for m in metrics], 0.36, color=COLORS[0], label="Schema valid"
)
axes[0].bar(
    x + 0.18, [m.macro_field_f1 for m in metrics], 0.36, color=COLORS[1], label="Macro field score"
)
axes[0].set(
    xticks=x,
    xticklabels=["Minimal", "Totals", "Line items"],
    ylim=(0, 1.16),
    title="Measured regex fixture extraction",
)
axes[0].legend(fontsize=9)
axes[1].bar(x, [m.line_item_f1 for m in metrics], color=COLORS[1])
axes[1].set(
    xticks=x,
    xticklabels=["Minimal", "Totals", "Line items"],
    ylim=(0, 1.1),
    title="Line matching includes quantity",
    ylabel="Line-item F1",
)
ranks = [1, 2, 4, 8, 16, 32, 64]
counts = [LoRALinear(512, 512, r=r).get_parameter_counts() for r in ranks]
fractions = [100 * c["trainable_fraction"] for c in counts]
axes[2].plot(ranks, fractions, "o-", color=COLORS[2])
axes[2].set(
    xscale="log",
    xticks=ranks,
    xticklabels=ranks,
    xlabel="Adapter rank r",
    ylabel="Trainable fraction (%)",
    title="LoRA: analytical 512 x 512 layer",
)
fig.suptitle("Receipt extraction | heuristic demo and exact LoRA parameter counts", fontsize=16)
fig.supxlabel(
    "Two synthetic test receipts; no local LLM inference. Parameter curve is algebra,"
    " not measured QLoRA GPU performance.",
    fontsize=10,
)
payload = {
    "source": "two synthetic test receipts; regex inference only",
    "conditions": conditions,
    "schema_validity": [m.schema_validity_rate for m in metrics],
    "macro_field_score": [m.macro_field_f1 for m in metrics],
    "line_item_f1": [m.line_item_f1 for m in metrics],
    "ranks": ranks,
    "trainable_fraction_pct": fractions,
}

source_hash = hashlib.sha256()
for source_path in sorted((ROOT / "src").rglob("*.py")):
    source_hash.update(str(source_path.relative_to(ROOT)).encode())
    source_hash.update(source_path.read_bytes())
payload["source_sha256"] = source_hash.hexdigest()

for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
fig.savefig(ROOT / "image.png", dpi=180, metadata={"Software": "Matplotlib"})
plt.close(fig)
(ROOT / "scripts" / "figure-data.json").write_text(
    json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)

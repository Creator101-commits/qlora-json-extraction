"""Render image.png from the curated part A results in results/.

Panels: (a) per-receipt test field F1 strips, (b) test field F1 with 95%
bootstrap CIs, (c) parse and schema-valid rates per condition.

Panel (a) rescores results/jobs/<condition>/predictions.jsonl with the
repo's own scorer (CORD-v2 gold labels via load_cord_split, predictions
via parse_model_output and score_doc). The reproduction gate recomputes
the micro field F1 per condition and aborts the render if it does not
match results/metrics.json to four decimals, so the figure can never
disagree with the published numbers.

Run from the repository root:

    uv run --with matplotlib python scripts/render_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from qlora_extraction.data.cord import load_cord_split  # noqa: E402
from qlora_extraction.evaluation.metrics import _f1, score_doc  # noqa: E402
from qlora_extraction.inference.parse import parse_model_output  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_PNG = ROOT / "image.png"

CONDITIONS = [
    ("zero_shot", "#9AA5B1", "Zero-shot"),
    ("few_shot", "#E9A23B", "Few-shot"),
    ("qlora_sft_seed0", "#3B6EA8", "QLoRA"),
]

INK = "#1F2A37"
BLUE = "#3B6EA8"
TEAL = "#2A9D8F"
GRAY = "#9AA5B1"
GRID = "#EEF0F2"
SPINE = "#B8BEC6"
TICK_LABEL = "#4A5361"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.8,
        "xtick.color": SPINE,
        "ytick.color": SPINE,
        "xtick.labelcolor": TICK_LABEL,
        "ytick.labelcolor": TICK_LABEL,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

FIGSIZE = (12, 3.84)
DPI = 150


def _fmt2(x: float, _pos: int) -> str:
    """Up to 2 decimals inside mathtext; large values as integers."""
    if x == 0:
        return "$0$"
    if abs(x) >= 1000:
        return f"${x:.0f}$"
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        return "$0$"
    return f"${s}$"


CATEGORICAL: list[tuple[plt.Axes, str, list[str]]] = []


def style(
    ax: plt.Axes,
    letter: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=11, color=INK)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_color(SPINE)
        ax.spines[name].set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.6,
        color=SPINE,
        labelcolor=TICK_LABEL,
        labelsize=8.5,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def cat(
    ax: plt.Axes,
    axis: str,
    positions: list[int] | range,
    labels: list[str],
) -> None:
    """Categorical ticks via FixedLocator + FixedFormatter, applied after
    style() so the numeric formatter cannot clobber the labels."""
    labels = list(labels)
    if axis == "x":
        ax.set_xticks(positions)
        ax.xaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
    else:
        ax.set_yticks(positions)
        ax.yaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
    CATEGORICAL.append((ax, axis, labels))


def verify_categories() -> None:
    bad = []
    for ax, axis, expected in CATEGORICAL:
        got = [
            t.get_text()
            for t in (ax.get_xticklabels() if axis == "x" else ax.get_yticklabels())
        ]
        if got != expected:
            bad.append((axis, expected, got))
    CATEGORICAL.clear()
    if bad:
        details = "; ".join(f"{a} expected={e} got={g}" for a, e, g in bad)
        raise RuntimeError(f"categorical tick labels mismatch: {details}")


def _all_texts(fig: plt.Figure) -> list[plt.Text]:
    items = []
    for ax in fig.axes:
        cand = [ax.title, ax.xaxis.label, ax.yaxis.label]
        cand += list(ax.texts)
        for axis in (ax.xaxis, ax.yaxis):
            try:
                ticks = axis._update_ticks()  # noqa: SLF001
            except Exception:
                ticks = []
            for tick in ticks:
                lab = tick.label1
                if lab.get_text():
                    cand.append(lab)
        leg = ax.get_legend()
        if leg is not None:
            cand += list(leg.get_texts())
        for t in cand:
            try:
                if t.get_text() and t.get_visible():
                    items.append(t)
            except Exception:
                pass
    return items


def check_text_overlap(fig: plt.Figure) -> list[str]:
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    boxes = []
    for t in _all_texts(fig):
        bb = t.get_window_extent(ren)
        boxes.append((t.get_text()[:40], bb))
    problems = []
    for i, (s1, b1) in enumerate(boxes):
        if b1.x0 < -1 or b1.y0 < -1 or b1.x1 > W + 1 or b1.y1 > H + 1:
            problems.append(f"OUT-OF-FIGURE {s1!r} bbox={b1}")
        for s2, b2 in boxes[i + 1 :]:
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > 1.5 and iy > 1.5:
                inter = ix * iy
                small = min(b1.width * b1.height, b2.width * b2.height)
                if inter > max(8.0, 0.10 * small):
                    problems.append(f"OVERLAP {s1!r} x {s2!r} area={inter:.0f}px2")
    return problems


def check_legend_data(fig: plt.Figure) -> list[str]:
    """Count data markers/vertices falling inside legend boxes."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(ren)
        covered = 0
        total = 0
        for coll in ax.collections:
            offs = coll.get_offsets()
            if len(offs) == 0 or len(offs) > 2000:
                continue
            pts = ax.transData.transform(offs)
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(offs)
        for line in ax.lines:
            if line.get_transform() != ax.transData:
                continue
            try:
                pts = ax.transData.transform(line.get_path().vertices)
            except Exception:
                continue
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(pts)
        for patch in ax.patches:
            try:
                pb = patch.get_window_extent(ren)
            except Exception:
                continue
            if pb.x0 < lb.x1 and pb.x1 > lb.x0 and pb.y0 < lb.y1 and pb.y1 > lb.y0:
                covered += 1
            total += 1
        if covered:
            out.append(f"legend covers ~{covered}/{total} data items")
    return out


def score_test_predictions(
    metrics: dict,
) -> dict[str, dict]:
    """Rescore test predictions with the repo's own scorer.

    Returns per-condition micro counts and per-document F1. Raises
    SystemExit when the reproduction gate fails.
    """
    test_docs = load_cord_split("test")
    by_id = {d.doc_id: d for d in test_docs}
    out: dict[str, dict] = {}
    gate: dict[str, dict] = {}
    for cond, _color, _label in CONDITIONS:
        tp = fp = fn = 0
        docs: dict[str, float] = {}
        n = 0
        path = RESULTS / "jobs" / cond / "predictions.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "test":
                continue
            doc = by_id.get(row["doc_id"])
            if doc is None:
                continue
            parsed = parse_model_output(row["raw_text"])
            s = score_doc(doc, parsed)
            tp += s.tp
            fp += s.fp
            fn += s.fn
            n += 1
            _p, _r, f1 = _f1(s.tp, s.fp, s.fn)
            docs[row["doc_id"]] = f1
        micro_f1 = _f1(tp, fp, fn)[2]
        expected = metrics["test"][cond]["field_f1"]
        expected_n = metrics["test"][cond]["n_docs"]
        ok = round(micro_f1, 4) == round(expected, 4) and n == expected_n
        gate[cond] = {
            "computed": round(micro_f1, 4),
            "expected": round(expected, 4),
            "n": n,
            "expected_n": expected_n,
            "ok": ok,
        }
        out[cond] = {"n": n, "micro_field_f1": micro_f1, "docs": docs}
    print("reproduction gate:", json.dumps(gate, indent=2))
    if not all(g["ok"] for g in gate.values()):
        sys.exit("reproduction gate failed; refusing to render image.png")
    return out


def render(metrics: dict, scores: dict[str, dict]) -> None:
    labels = [label for _c, _col, label in CONDITIONS]
    fig, (a1, a2, a3) = plt.subplots(
        1, 3, figsize=FIGSIZE, dpi=DPI, layout="constrained"
    )

    rng = np.random.default_rng(0)
    for i, (cond, color, _label) in enumerate(CONDITIONS):
        vals = np.array(list(scores[cond]["docs"].values()))
        jit = rng.uniform(-0.16, 0.16, size=len(vals))
        a1.scatter(i + jit, vals, s=14, color=color, alpha=0.55,
                   edgecolors="none", zorder=3)
    a1.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    a1.set_ylim(-0.02, 1.02)
    style(a1, "a", "Per-receipt field F1 (test, n = 100)",
          ylabel="Field F1")
    cat(a1, "x", range(len(CONDITIONS)), labels)

    boot = metrics["bootstrap"]["conditions"]
    for i, (cond, color, _label) in enumerate(CONDITIONS):
        f1 = metrics["test"][cond]["field_f1"]
        ci = boot[cond]["field_f1"]
        a2.errorbar(
            [i],
            [f1],
            yerr=[[f1 - ci["ci_low"]], [ci["ci_high"] - f1]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            ms=7,
            markeredgecolor=color,
            markerfacecolor=color,
        )
    a2.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    a2.set_ylim(0, 1)
    style(a2, "b", "Field F1 with 95% bootstrap CI",
          ylabel="Field F1")
    cat(a2, "x", range(len(CONDITIONS)), labels)

    width = 0.36
    for i, (cond, _color, _label) in enumerate(CONDITIONS):
        t = metrics["test"][cond]
        a3.bar(i - width / 2 - 0.02, t["parse_rate"], width,
               color=BLUE, label="Parse rate" if i == 0 else None)
        a3.bar(i + width / 2 + 0.02, t["schema_valid_rate"], width,
               color=TEAL, label="Schema-valid rate" if i == 0 else None)
    a3.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    a3.set_ylim(0, 1.12)
    style(a3, "c", "Format compliance (test)", ylabel="Rate")
    cat(a3, "x", range(len(CONDITIONS)), labels)
    a3.legend(loc="upper center", ncol=2)

    fig.canvas.draw()
    fig.canvas.draw()
    verify_categories()
    fig.savefig(OUT_PNG, dpi=DPI, facecolor="white")
    for kind, problems in (
        ("text", check_text_overlap(fig)),
        ("legend-data", check_legend_data(fig)),
    ):
        for p in problems:
            print(f"  CHECK {kind}: {p}")
    plt.close(fig)
    print("wrote", OUT_PNG)


def main() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    scores = score_test_predictions(metrics)
    render(metrics, scores)


if __name__ == "__main__":
    main()

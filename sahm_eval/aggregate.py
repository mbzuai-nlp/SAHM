"""Aggregate per-run task results into mean+-std tables and a leaderboard.

MCQ tasks are reported as accuracy (%) and generative tasks as the judge's
0-10 `overall` mean. The two are NOT mixed into a single number — they live in
separate tables, matching the original SAHM result tables.
"""

import json
import statistics
from datetime import datetime
from pathlib import Path


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def _table(title, unit, tasks, summary):
    if not tasks:
        return []
    md = [f"## {title}\n",
          "| Rank | Model | " + " | ".join(tasks) + f" | **Avg ({unit})** |",
          "|---|---|" + "|".join(["---"] * len(tasks)) + "|---|"]
    rows = []
    for model, per_task in summary.items():
        cells, means = [], []
        for t in tasks:
            s = per_task.get(t)
            if s and s["runs"]:
                cells.append(f"{s['mean']:.2f}±{s['std']:.2f}")
                means.append(s["mean"])
            else:
                cells.append("—")
        avg = statistics.mean(means) if means else None
        rows.append((avg if avg is not None else -1, model, cells, avg))
    for i, (_, model, cells, avg) in enumerate(sorted(rows, reverse=True), 1):
        avg_s = f"{avg:.2f}" if avg is not None else "—"
        md.append(f"| {i} | {model} | " + " | ".join(cells) + f" | **{avg_s}** |")
    md.append("")
    return md


def build_leaderboard(per_run: dict, task_types: dict, out_dir: Path):
    """
    per_run    : {model: {task: [score_run1, score_run2, ...]}}
    task_types : {task: "mcq" | "generative"}
    Writes leaderboard.md / leaderboard.json; returns the per-model summary.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = sorted({t for m in per_run.values() for t in m})
    mcq_tasks = [t for t in all_tasks if task_types.get(t) == "mcq"]
    gen_tasks = [t for t in all_tasks if task_types.get(t) == "generative"]

    summary = {}
    for model, task_scores in per_run.items():
        row = {}
        for task in all_tasks:
            mean, std = mean_std(task_scores.get(task, []))
            row[task] = {"mean": round(mean, 2) if mean is not None else None,
                         "std": round(std, 2) if std is not None else None,
                         "runs": len(task_scores.get(task, []))}
        summary[model] = row

    md = [f"# SAHM Leaderboard", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_\n"]
    md += _table("MCQ — accuracy (%)", "%", mcq_tasks, summary)
    md += _table("Generative — LLM-judge overall (0–10)", "/10", gen_tasks, summary)
    (out_dir / "leaderboard.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "leaderboard.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

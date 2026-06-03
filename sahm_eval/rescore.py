"""Offline re-scoring of existing MCQ generation files (no GPU, no judge)."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

from . import mcq


def _infer_num_choices(rec: dict) -> int:
    ch = rec.get("choices")
    if isinstance(ch, (dict, list, tuple)) and ch:
        return len(ch)
    return 5


def score_file(path: Path) -> dict:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records.append(mcq.score_record(rec, _infer_num_choices(rec)))
    return mcq.accuracy(records)


def score_tree(root: Path):
    files = sorted(root.rglob("*generations.jsonl"))
    groups = defaultdict(list)
    for fp in files:
        dataset = fp.parent.name
        model = "unknown"
        for anc in fp.parents:
            if anc.name.startswith(("run_2025", "run_2026")):
                model = anc.name.split("_", 3)[-1]
                break
        groups[(model, dataset)].append(score_file(fp)["accuracy"])
    return len(files), groups


def run(args):
    if args.path:
        stats = score_file(Path(args.path))
        print(f"{Path(args.path).name}: {stats['accuracy']:.2f}%  "
              f"({stats['correct']}/{stats['total']}, invalid={stats['invalid']})")
        return
    n, groups = score_tree(Path(args.tree))
    print(f"Found {n} generation files under {args.tree}\n")
    print(f"{'MODEL':32} {'DATASET':24} {'ACC (mean+-std)':>18} {'runs':>5}")
    print("-" * 84)
    for (model, dataset), accs in sorted(groups.items()):
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"{model:32.32} {dataset:24.24} {mean:8.2f} +- {std:5.2f}    {len(accs):>5}")

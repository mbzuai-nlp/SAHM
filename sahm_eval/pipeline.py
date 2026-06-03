"""Evaluation pipeline: load tasks -> generate -> score -> leaderboard."""

import json
from datetime import datetime
from pathlib import Path

from . import mcq
from .config import load_tasks, load_models
from .loader import load_task
from .aggregate import build_leaderboard

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# Generative tasks with a verbatim judge rubric. Summarization had no LLM-judge
# in the original pipeline, so it is generation-only.
JUDGED_RUBRICS = {"financial_qa", "fatwa_qa", "islamic_qa"}


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(args) -> dict:
    registry = load_tasks(args.tasks_file)
    open_models, api_models = load_models(args.models_file)

    task_keys = list(registry) if "all" in args.tasks else args.tasks
    for t in task_keys:
        if t not in registry:
            raise SystemExit(f"unknown task '{t}'. Available: {list(registry)}")

    seeds = (args.seeds or DEFAULT_SEEDS)[:args.runs]
    while len(seeds) < args.runs:
        seeds.append(seeds[-1] + 1000)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) / f"run_{stamp}_{args.model}"

    # --- backend (heavy imports kept lazy) ---
    if args.backend == "vllm":
        if args.model not in open_models:
            raise SystemExit(f"'{args.model}' not in open_models. Available: {list(open_models)}")
        from .vllm_runner import VLLMModel
        model = VLLMModel(open_models[args.model])
    else:
        if args.model not in api_models:
            raise SystemExit(f"'{args.model}' not in api_models. Available: {list(api_models)}")
        from .api_runner import APIModel
        model = APIModel(api_models[args.model])

    needs_judge = any(registry[t]["type"] == "generative"
                      and registry[t].get("rubric") in JUDGED_RUBRICS for t in task_keys)
    judge = None
    if needs_judge and not args.skip_judge:
        from .judge import Judge
        judge = Judge(model=args.judge_model, base_url=args.judge_base_url,
                      api_key_env=args.judge_key_env)

    per_run = {args.model: {}}
    task_types = {}

    for run_idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}\nRUN {run_idx}/{args.runs}  (seed={seed})\n{'='*60}")
        run_dir = out_root / f"run_{run_idx}_seed{seed}"

        for tkey in task_keys:
            cfg = registry[tkey]
            disp = cfg["display_name"]
            records, split = load_task(cfg, limit=args.limit)
            task_types[disp] = cfg["type"]
            print(f"  [{disp}] {cfg['type']} | {split} | {len(records)} ex")

            if cfg["type"] == "mcq":
                gens = model.run_mcq(records, max_new_tokens=cfg.get("max_new_tokens", 8), seed=seed)
                scored = [mcq.score_record(g, cfg["num_choices"], g.get("first_token_logprobs"))
                          for g in gens]
                stats = mcq.accuracy(scored)
                for g, s in zip(gens, scored):
                    g.pop("first_token_logprobs", None)
                    g["_score"] = s
                _write_jsonl(run_dir / disp / "generations.jsonl", gens)
                (run_dir / disp / "score.json").write_text(json.dumps(stats, indent=2))
                per_run[args.model].setdefault(disp, []).append(stats["accuracy"])
                print(f"     accuracy = {stats['accuracy']:.2f}%  (invalid {stats['invalid']})")
            else:
                gens = model.run_generative(records, max_new_tokens=cfg.get("max_new_tokens", 1024), seed=seed)
                _write_jsonl(run_dir / disp / "generations.jsonl", gens)
                rubric = cfg.get("rubric")
                if judge and rubric in JUDGED_RUBRICS:
                    from .judge import aggregate_scores
                    jr = judge.score_records(gens, rubric)
                    agg = aggregate_scores(jr)
                    _write_jsonl(run_dir / disp / "judge.jsonl", jr)
                    (run_dir / disp / "score.json").write_text(json.dumps(agg, indent=2))
                    per_run[args.model].setdefault(disp, []).append(agg["mean"])
                    print(f"     judge = {agg['mean']:.2f}/10  (n={agg['n']}, failed={agg['n_failed']})")
                else:
                    why = "no judge rubric" if rubric not in JUDGED_RUBRICS else "judge skipped"
                    print(f"     (generation only; {why})")

    summary = build_leaderboard(per_run, task_types, out_root)
    print(f"\nDone. Outputs in {out_root}")
    print(f"Leaderboard: {out_root / 'leaderboard.md'}")
    return summary

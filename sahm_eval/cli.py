"""``sahm-eval`` command-line interface.

    sahm-eval run        --model NAME --tasks all --runs 3
    sahm-eval score-mcq  --tree results/        # re-score generations offline
    sahm-eval list                              # show available tasks & models
"""

import argparse

from . import __version__
from .config import load_tasks, load_models


def _add_run(sub):
    p = sub.add_parser("run", help="evaluate a model on SAHM tasks")
    p.add_argument("--model", required=True, help="model key (see `sahm-eval list`)")
    p.add_argument("--backend", choices=["vllm", "api"], default="vllm",
                   help="vllm = local GPU inference; api = hosted OpenAI-compatible model")
    p.add_argument("--tasks", nargs="+", default=["all"],
                   help="task keys, or 'all' (default)")
    p.add_argument("--runs", type=int, default=1, help="number of seeded runs")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="explicit seeds")
    p.add_argument("--limit", type=int, default=None, help="cap examples/task (debug)")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--tasks-file", default=None, help="override packaged tasks.yaml")
    p.add_argument("--models-file", default=None, help="override packaged models.yaml")
    p.add_argument("--judge-model", default="gpt-4o", help="LLM-judge model for generative tasks")
    p.add_argument("--judge-base-url", default=None, help="OpenAI-compatible base URL for the judge")
    p.add_argument("--judge-key-env", default="OPENAI_API_KEY", help="env var holding the judge API key")
    p.add_argument("--skip-judge", action="store_true", help="generate only; score generative later")


def _add_score(sub):
    p = sub.add_parser("score-mcq", help="re-score existing MCQ generations (no GPU, no judge)")
    p.add_argument("path", nargs="?", help="a single *_generations.jsonl file")
    p.add_argument("--tree", help="a results dir to recurse and aggregate")


def _add_list(sub):
    sub.add_parser("list", help="list available tasks and models")
    sub.add_parser("version", help="print version")


def _cmd_list(args):
    tasks = load_tasks(args.tasks_file if hasattr(args, "tasks_file") else None)
    open_m, api_m = load_models(None)
    print("Tasks:")
    for k, c in tasks.items():
        print(f"  {k:16} {c['type']:11} {c['display_name']}")
    print("\nOpen models (vllm backend):")
    for k in open_m:
        print(f"  {k}")
    print("\nAPI models (api backend):")
    for k in api_m:
        print(f"  {k}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sahm-eval",
        description="SAHM — Arabic financial & Shari'ah-compliant benchmark evaluation harness.")
    parser.add_argument("-V", "--version", action="version", version=f"sahm-eval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run(sub)
    _add_score(sub)
    _add_list(sub)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run":
        from . import pipeline
        pipeline.run(args)
    elif args.command == "score-mcq":
        if not args.path and not args.tree:
            raise SystemExit("score-mcq needs a file path or --tree DIR")
        from . import rescore
        rescore.run(args)
    elif args.command == "list":
        args.tasks_file = None
        _cmd_list(args)
    elif args.command == "version":
        print(__version__)


if __name__ == "__main__":
    main()

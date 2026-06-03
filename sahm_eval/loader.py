"""Load and normalise SAHM tasks from the Hub into a common record shape."""

import os


def _pick_split(ds_dict):
    for s in ("test", "validation", "train"):
        if s in ds_dict:
            return ds_dict[s], s
    first = list(ds_dict.keys())[0]
    return ds_dict[first], first


def load_task(task_cfg: dict, limit: int = None, hf_token: str = None):
    """
    Return (records, split_name). Each record is a normalised dict:

      mcq        -> {id, prompt, answer, gold, num_choices}
      generative -> {id, prompt, reference}
    """
    from datasets import load_dataset  # heavy import, kept lazy
    token = hf_token or os.getenv("HF_TOKEN")
    ds_dict = load_dataset(task_cfg["hf_path"], token=token)
    ds, split = _pick_split(ds_dict)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    records = []
    if task_cfg["type"] == "mcq":
        pc, ac, gc = task_cfg["prompt_col"], task_cfg["answer_col"], task_cfg.get("gold_index_col")
        n = task_cfg["num_choices"]
        for i, row in enumerate(ds):
            records.append({
                "id": row.get("id", f"{task_cfg['display_name']}_{i}"),
                "prompt": row.get(pc, ""),
                "answer": row.get(ac),
                "gold": row.get(gc) if gc else None,
                "num_choices": n,
            })
    else:
        pc, rc = task_cfg["prompt_col"], task_cfg["reference_col"]
        for i, row in enumerate(ds):
            records.append({
                "id": row.get("id", f"{task_cfg['display_name']}_{i}"),
                "prompt": row.get(pc, ""),
                "reference": row.get(rc, ""),
                # carried through for the fatwa judge prompt (optional context)
                "category": row.get("category"),
                "topic": row.get("topic"),
            })
    return records, split

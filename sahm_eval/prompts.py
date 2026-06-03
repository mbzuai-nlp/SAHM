"""Shared prompt/stop constants (no heavy deps, safe to import anywhere)."""

# EXACT original instruction from infer_mcq.py — do not change (replication).
MCQ_INSTRUCTION = "أجب على السؤال التالي باختيار الحرف الصحيح فقط (a, b, c، أو d):"

STOP = ["<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>",
        "<end_of_turn>", "</s>", "<|assistant|>", "[/INST]"]

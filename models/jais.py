# -*- coding: utf-8 -*-
# ar_fiqh_eval_jais_vllm_letter_only.py
"""
Evaluate **Jais** (via vLLM) on Arabic fiqh MCQs in **letter-only** mode.

Input JSONL (one per line):
{
  "question": "...",
  "options": {"A":"...", "B":"...", "C":"...", "D":"..."},
  "correct_answer": "A",
  "explanation": "..."   # ignored in letter-only mode
}

Usage (deterministic, batched):
  pip install vllm
  python jais.py \
    --input ../data/islamic_finance/fatwa_mcq/pure_fatwa_mcq.jsonl \
    --model inceptionai/jais-family-6p7b-chat \
    --output_dir runs/jais_vllm_letter \
    --max_new_tokens 8 --temperature 0.0 --top_p 1.0 --batch_size 32
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from typing import List

from vllm import LLM, SamplingParams

# ---------------- Prompt template (Jais chat) ----------------

PROMPT_AR = (
    "### Instruction: اسمك \"جيس\" وسُمّيت على اسم جبل جيس أعلى جبل في الإمارات. "
    "تمَّ بناؤك بواسطة Inception في الإمارات. أنت مساعدٌ مفيد ومحترم وصادق. "
    "أجب دائمًا بأكبر قدر من المساعدة مع الالتزام بالسلامة. "
    "أكمِل المحادثة بين [|Human|] و[|AI|]:\n"
    "### Input: [|Human|] {Question}\n[|AI|]\n### Response :"
)

TAIL_LETTER_ONLY = " أجب بحرف الخيار فقط (A/B/C/D) دون شرح."

LETTER_RE = re.compile(r"\b([A-D])\b")


def load_data(path: str) -> List[dict]:
    p = Path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_mcq_prompt(ex: dict) -> str:
    q = ex["question"]
    A = ex["options"]["A"]
    B = ex["options"]["B"]
    C = ex["options"]["C"]
    D = ex["options"]["D"]
    mcq = (
        f"سؤال: {q}\n\nالخيارات:\n"
        f"A) {A}\nB) {B}\nC) {C}\nD) {D}\n\n{TAIL_LETTER_ONLY}"
    )
    return PROMPT_AR.format(Question=mcq)


def extract_letter(text: str) -> str:
    """
    Extract first occurrence of A/B/C/D.
    Accepts outputs that may or may not include the '### Response :' header.
    """
    if "### Response :" in text:
        text = text.split("### Response :", 1)[-1].strip()
    elif "### Response:" in text:
        text = text.split("### Response:", 1)[-1].strip()
    m = LETTER_RE.search(text)
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="ar_fiqh_eval.jsonl")
    ap.add_argument("--model", type=str, default="inceptionai/jais-family-6p7b-chat")
    ap.add_argument("--output_dir", type=str, default="runs/jais_vllm_letter")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "half", "bfloat16", "float32"])
    ap.add_argument("--swap_space", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_data(args.input)

    # Build prompts (letter-only)
    prompts = [build_mcq_prompt(ex) for ex in rows]

    # vLLM engine
    llm = LLM(
        model=args.model,
        trust_remote_code=True,              # Jais chat uses custom template code
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,                    # let vLLM choose if "auto"
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
    )

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        # Stop if the model starts a new section or restarts the prompt
        stop=["\n###", "### Instruction:", "### Input:", "[|Human|]"],
    )

    # Batched generation
    raw_outputs = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        results = llm.generate(batch, sampling)
        for res in results:
            txt = res.outputs[0].text if res.outputs else ""
            raw_outputs.append(txt)

    # Scoring (accuracy only)
    out_csv = Path(args.output_dir) / "per_item.csv"
    summary = Path(args.output_dir) / "summary.txt"
    correct = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "gold", "pred", "is_correct", "raw_output"])
        for idx, (ex, raw) in enumerate(zip(rows, raw_outputs)):
            pred = extract_letter(raw)
            is_corr = int(pred == ex["correct_answer"])
            correct += is_corr
            w.writerow([idx, ex["correct_answer"], pred, is_corr, raw])

    acc = correct / max(1, len(rows))
    with summary.open("w", encoding="utf-8") as f:
        f.write(f"Items: {len(rows)}\nAccuracy: {acc:.3f}\n")
        f.write("Notes:\n- Mode: LETTER-ONLY (A/B/C/D)\n- Provider: Jais via vLLM\n")

    print(f"Done. Accuracy={acc:.3f}")
    print(f"Per-item: {out_csv}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()

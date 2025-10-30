"""
Evaluator for the Arabic Financial QA dataset.

- Takes a model predictions file (CSV/JSONL/Parquet) with columns: id, split, model_answer
- Merges with ground-truth data (SahmBenchmark/arabic-financial-qa_eval) by id
- For each example, builds an evaluation rubric prompt and scores the model answer
- Produces structured outputs with subscores, overall score (out of 10), and notes
- Saves results as JSONL, Parquet, and CSV
"""

from __future__ import annotations
import os
import re
import json
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import pandas as pd
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from google.ai.generativelanguage_v1beta.types import content  # for Schema

# --------------------------- Config ---------------------------

@dataclass
class JudgeConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_output_tokens: int = 4096  # higher to avoid truncation
    note_language: str = "Arabic"  # "Arabic" or "English"
    response_mime_type: str = "application/json"

# --------------------------- Rubric ---------------------------

RUBRIC_EN = """
You are an expert evaluator in financial analysis and capital markets. You will be given, each time:
- prompt – the full Arabic prompt (report/excerpt + question) that the model saw
- ground_truth – the reference ideal analytical answer, in Arabic
- candidate_answer – the model’s answer to be evaluated, in Arabic

Your task:
- Judge how well the candidate_answer matches the ground_truth in *conclusions*, *reasoning*, and *use of provided figures*.
- Prioritize factual/quantitative fidelity, correct interpretation of financial concepts (e.g., spreads, yields, coverage, issuance, capital structure, Basel III, supply/demand dynamics), and avoidance of hallucinated data.
- Do not penalize stylistic paraphrase if core insights and numeric takeaways align with the ground_truth.

Score on the following criteria so that the total sums to exactly 10:

1) Core conclusion alignment (0–4):
   - Does the candidate capture the main thesis and key takeaways of the ground_truth (what/why/so-what)?

2) Quantitative fidelity & use of figures (0–2):
   - Correctly cites or uses the *reported* numbers (e.g., percentages, amounts, maturities, oversubscription) without inventing/altering figures.
   - Any simple computations or comparisons are consistent.

3) Financial reasoning soundness (0–2):
   - Causality and mechanisms are plausible and consistent with standard finance/econ logic (e.g., pricing vs. credit risk, duration/tenor structure, demand/oversubscription signals, capital adequacy).

4) Clarity & Arabic language quality (0–1):
   - Clear Arabic, coherent structure, minimal ambiguity.

5) Directness & on-topic grounding (0–1):
   - Answers what was asked, stays anchored in the provided scenario/data (no generic filler).

Additionally, set the following critical checks (true/false):

- contradicts_ground_truth: Does the candidate contradict the central conclusion of the ground_truth?
- fabricates_or_alters_numbers: Does it introduce numbers not present or distort/alter reported figures materially?
- hallucinates_context_or_sources: Does it inject external context/sources not in the prompt that change the assessment?
- flawed_financial_logic: Is there a serious finance/econ reasoning error that would mislead the conclusion?
- non_answer_or_evasive: Does it avoid providing an analytical answer to the question?
- off_topic_or_unsafe: Is it off-topic or otherwise inappropriate?

Important instructions:
- Compare to ground_truth: we want alignment with the reference meaning and key numeric/analytic points.
- Minor wording differences are fine if substance matches.
- Keep the final note concise (one or two sentences).
- Output *only* valid JSON, no prose, no code fences.

Return JSON *strictly* in this schema (all fields required):

{
  "scores": {
    "coverage_core_conclusion": <float 0-4>,
    "quantitative_fidelity": <float 0-2>,
    "financial_reasoning": <float 0-2>,
    "clarity_language": <float 0-1>,
    "directness_grounding": <float 0-1>
  },
  "overall": <float 0-10>,  // must equal exact sum of the five subscores
  "critical_checks": {
    "contradicts_ground_truth": <true/false>,
    "fabricates_or_alters_numbers": <true/false>,
    "hallucinates_context_or_sources": <true/false>,
    "flawed_financial_logic": <true/false>,
    "non_answer_or_evasive": <true/false>,
    "off_topic_or_unsafe": <true/false>
  },
  "note": "<short NOTE in {NOTE_LANG}>"
}
""".strip()

# --------------------------- JSON Schema for Gemini ---------------------------

def build_response_schema() -> content.Schema:
    return content.Schema(
        type=content.Type.OBJECT,
        properties={
            "scores": content.Schema(
                type=content.Type.OBJECT,
                properties={
                    "coverage_core_conclusion": content.Schema(type=content.Type.NUMBER),
                    "quantitative_fidelity": content.Schema(type=content.Type.NUMBER),
                    "financial_reasoning": content.Schema(type=content.Type.NUMBER),
                    "clarity_language": content.Schema(type=content.Type.NUMBER),
                    "directness_grounding": content.Schema(type=content.Type.NUMBER),
                },
                required=[
                    "coverage_core_conclusion",
                    "quantitative_fidelity",
                    "financial_reasoning",
                    "clarity_language",
                    "directness_grounding",
                ],
            ),
            "overall": content.Schema(type=content.Type.NUMBER),
            "critical_checks": content.Schema(
                type=content.Type.OBJECT,
                properties={
                    "contradicts_ground_truth": content.Schema(type=content.Type.BOOLEAN),
                    "fabricates_or_alters_numbers": content.Schema(type=content.Type.BOOLEAN),
                    "hallucinates_context_or_sources": content.Schema(type=content.Type.BOOLEAN),
                    "flawed_financial_logic": content.Schema(type=content.Type.BOOLEAN),
                    "non_answer_or_evasive": content.Schema(type=content.Type.BOOLEAN),
                    "off_topic_or_unsafe": content.Schema(type=content.Type.BOOLEAN),
                },
                required=[
                    "contradicts_ground_truth",
                    "fabricates_or_alters_numbers",
                    "hallucinates_context_or_sources",
                    "flawed_financial_logic",
                    "non_answer_or_evasive",
                    "off_topic_or_unsafe",
                ],
            ),
            "note": content.Schema(type=content.Type.STRING),
        },
        required=["scores", "overall", "critical_checks", "note"],
    )

# --------------------------- Utility ---------------------------

def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty judge response text.")
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in judge response.")
    return json.loads(m.group(0))

def try_repair_json(raw: str) -> Dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.replace("\uFEFF", "")
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        return extract_json(cleaned)
    except Exception:
        pass
    cleaned2 = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return json.loads(cleaned2)

def read_predictions(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported predictions file type: {ext}")
    needed = ["id", "split", "model_answer"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Predictions file missing required column '{c}'")
    return df

def load_ground_truth(split: str) -> pd.DataFrame:
    """
    Expected dataset columns: id, prompt, answer
    """
    ds = load_dataset("SahmBenchmark/arabic-financial-qa_eval", split=split)
    df = ds.to_pandas()
    for col in ["id", "prompt", "answer"]:
        if col not in df.columns:
            raise ValueError(f"Ground-truth missing required column: {col}")
    keep_cols = ["id", "prompt", "answer"]
    return df[keep_cols]

# --------------------------- Judge class ---------------------------

class GeminiJudge:
    def __init__(self, api_key: str, cfg: JudgeConfig):
        self.cfg = cfg
        genai.configure(api_key=api_key)

        # Relax safety to reduce spurious blocks on Arabic finance/news content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        self.schema = build_response_schema()

        self.model = genai.GenerativeModel(
            model_name=cfg.model,
            system_instruction=(
                "You are a precise, rigorous evaluator for Arabic financial QA. "
                "Score deterministically and output only valid JSON."
            ),
            generation_config={
                "temperature": cfg.temperature,
                "max_output_tokens": cfg.max_output_tokens,
                "response_mime_type": cfg.response_mime_type,
                "response_schema": self.schema,
            },
            safety_settings=self.safety_settings,
        )

    def create_evaluation_prompt(self, prompt_text: str, ground_truth: str, model_response: str) -> str:
        note_lang_str = "Arabic" if self.cfg.note_language.lower().startswith("arab") else "English"
        rubric = RUBRIC_EN.replace("{NOTE_LANG}", note_lang_str)
        parts = [
            rubric,
            "\n---\n",
            "The following fields are in ARABIC (do not translate them; evaluate against each other):",
            "Note: 'prompt' already includes the financial report/excerpt and the question.",
            f"prompt (AR):\n{prompt_text.strip()}",
            f"ground_truth (AR):\n{ground_truth.strip()}",
            f"candidate_answer (AR):\n{model_response.strip()}",
            "\nReturn ONLY JSON now.\n"
        ]
        return "\n".join(parts)

    def _extract_text(self, resp) -> str:
        if hasattr(resp, "candidates") and resp.candidates:
            cand = resp.candidates[0]
            if hasattr(cand, "content") and getattr(cand.content, "parts", None):
                txts = []
                for p in cand.content.parts:
                    if hasattr(p, "text") and p.text:
                        txts.append(p.text)
                if txts:
                    return "\n".join(txts).strip()
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        return ""

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
    )
    def judge_one(self, prompt: str) -> Dict[str, Any]:
        resp = self.model.generate_content(prompt)
        text_out = self._extract_text(resp)
        if not text_out:
            fr = None
            try:
                fr = resp.candidates[0].finish_reason
            except Exception:
                pass
            raise ValueError(f"Empty judge response. finish_reason={fr!r}")
        try:
            return extract_json(text_out)
        except Exception:
            return try_repair_json(text_out)

# --------------------------- Main Runner ---------------------------

def run(
    preds_path: str,
    split: str,
    out_dir: str,
    limit: Optional[int],
    cfg: JudgeConfig,
    api_key: Optional[str],
):
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini API key not provided. Set GEMINI_API_KEY or pass --api-key.")

    os.makedirs(out_dir, exist_ok=True)

    preds = read_predictions(preds_path)
    if limit:
        preds = preds.head(limit).copy()

    gt = load_ground_truth(split)

    df = preds.merge(gt, on="id", how="inner", suffixes=("", "_gt"))
    if df.empty:
        raise RuntimeError("No matching IDs between predictions and ground-truth.")

    judge = GeminiJudge(api_key=api_key, cfg=cfg)

    rows: List[Dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
        prompt_text = row.get("prompt", "")
        ground_truth = row.get("answer", "")
        candidate = row.get("model_answer", "")

        prompt = judge.create_evaluation_prompt(
            prompt_text=prompt_text,
            ground_truth=ground_truth,
            model_response=candidate,
        )

        try:
            result = judge.judge_one(prompt)
        except Exception as e:
            result = {
                "scores": {
                    "coverage_core_conclusion": 0.0,
                    "quantitative_fidelity": 0.0,
                    "financial_reasoning": 0.0,
                    "clarity_language": 0.0,
                    "directness_grounding": 0.0,
                },
                "overall": 0.0,
                "critical_checks": {
                    "contradicts_ground_truth": True,
                    "fabricates_or_alters_numbers": True,
                    "hallucinates_context_or_sources": False,
                    "flawed_financial_logic": True,
                    "non_answer_or_evasive": True,
                    "off_topic_or_unsafe": False,
                },
                "note": f"Judge error: {type(e).__name__}: {str(e)}",
                "_error": True,
            }

        record = {
            "id": row["id"],
            "split": split,
            "prompt": prompt_text,
            "ground_truth": ground_truth,
            "model_answer": candidate,
            "judge_scores": result.get("scores", {}),
            "judge_overall": result.get("overall", None),
            "judge_critical_checks": result.get("critical_checks", {}),
            "judge_note": result.get("note", ""),
            "judge_model": cfg.model,
            "judge_timestamp_utc": time.time(),
        }
        rows.append(record)

    out_df = pd.DataFrame(rows)

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    base = f"{split}__gemini25flash_financial__judged__{stamp}"

    jsonl_path = os.path.join(out_dir, f"{base}.jsonl")
    parquet_path = os.path.join(out_dir, f"{base}.parquet")
    csv_path = os.path.join(out_dir, f"{base}.csv")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, r in out_df.iterrows():
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    out_df.to_parquet(parquet_path, index=False)
    out_df.to_csv(csv_path, index=False)

    print("Saved judgments to:")
    print("  JSONL  :", jsonl_path)
    print("  Parquet:", parquet_path)
    print("  CSV    :", csv_path)

def main():
    parser = argparse.ArgumentParser(description="Gemini 2.5 Flash judge for Arabic Financial QA.")
    parser.add_argument("--preds-path", type=str, required=True, help="Path to predictions file (csv/jsonl/parquet).")
    parser.add_argument("--split", type=str, choices=["validation", "test"], required=True)
    parser.add_argument("--out-dir", type=str, default="research/Finance/outputs/judgments")
    parser.add_argument("--limit", type=int, default=None, help="Limit for dry-runs.")

    parser.add_argument("--judge-model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--judge-temp", type=float, default=0.0)
    parser.add_argument("--judge-max-output", type=int, default=4096)
    parser.add_argument("--note-language", type=str, default="Arabic", choices=["Arabic", "English"])
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API key (else read GEMINI_API_KEY).")

    args = parser.parse_args()

    cfg = JudgeConfig(
        model=args.judge_model,
        temperature=args.judge_temp,
        max_output_tokens=args.judge_max_output,
        note_language=args.note_language,
    )

    run(
        preds_path=args.preds_path,
        split=args.split,
        out_dir=args.out_dir,
        limit=args.limit,
        cfg=cfg,
        api_key=args.api_key,
    )

if __name__ == "__main__":
    main()



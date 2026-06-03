"""
Reference-based LLM-as-judge for the generative tasks.

IMPORTANT — REPLICATION:
The rubrics, system prompts, prompt assembly, JSON schema and the 0-10 `overall`
score below are COPIED VERBATIM from the original judge scripts
(rebuttal_evaluation/llm_judge_gpt/run_eval_*_gpt.py). Do NOT edit the prompt
text — it is what produced the published SAHM numbers. The rubric is identical to
the Gemini-judge version; only the backend model differs, so this same code
reproduces either the GPT or the Gemini judge depending on `model`.

Summarization is intentionally NOT judged here (it had no LLM-judge rubric in the
original pipeline) — run.py leaves it generation-only.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# ============================ VERBATIM RUBRICS ==============================

RUBRIC_FINANCIAL = """
You are an expert evaluator in financial analysis and capital markets. You will be given, each time:
- prompt – the full Arabic prompt (report/excerpt + question) that the model saw
- ground_truth – the reference ideal analytical answer, in Arabic
- candidate_answer – the model's answer to be evaluated, in Arabic

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
  "overall": <float 0-10>,
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

RUBRIC_FATWA = """
You are an expert evaluator in Islamic fatwa (iftaʾ). You will be given, each time:
- category (optional context) – may be empty (e.g., riba, zakat, takaful)
- prompt – the full Arabic prompt shown to the model (instructions + question)
- ground_truth – the reference fatwa answer, in Arabic
- candidate_answer – the model's fatwa answer to be evaluated, in Arabic

Your task:
- Judge how well the candidate_answer matches the ground_truth in *ruling (ḥukm)*, *justification* and *operative constraints/qualifications*.
- Prioritize doctrinal correctness and required conditions/exceptions. Rationale may be shorter or differently worded, but it must not change the ruling.
- Do not penalize stylistic paraphrase if the core ruling and constraints are preserved.
- Be concise and deterministic.

Score on the following criteria so that the total sums to exactly 10:

1) Coverage of core ruling (0–4):
   - The candidate must clearly state the same central hukm (e.g., validity/invalidity, permissibility/prohibition, enforceability) AND provide the key justification or reasoning present in the ground_truth.
   - One-word or minimal answers without the essential justification should receive a much lower score (e.g., 0–1).

2) Conditions, exceptions, constraints (0–2):
   - Does it retain critical restrictions, qualifiers, or carve-outs that materially affect the ruling?

3) Doctrinal/factual accuracy (0–2):
   - No misstatements that would change the fatwa, no implicit legalization of prohibited elements (e.g., ribā), no misleading generalizations or invented requirements.

4) Clarity & Arabic language quality (0–1):
   - Clear Arabic, understandable structure, minimal ambiguity appropriate for a fatwa answer.

5) Directness & fatwa format (0–1):
   - Directly answers the question; avoids long digressions; phrasing suitable for a fatwa.

Additionally, set the following critical checks (true/false):

- contradicts_ground_truth: Does the candidate contradict the central ruling?
- omits_critical_conditions: Does it omit key conditions/exceptions that change the ruling?
- introduces_unlawful_elements: Does it introduce/normalize prohibited elements (e.g., ribā)?
- hallucinated_citations: Misleading/fabricated sources claimed that distort the ruling?
- non_answer_or_evasive: Does it avoid giving a clear ruling?
- off_topic_or_unsafe: Off-topic or otherwise inappropriate?

Important:
- Compare to ground_truth: alignment with ruling and constraints.
- Keep the final note concise (one or two sentences).
- Output *only* valid JSON, no prose, no code fences.

Return JSON strictly in this schema:

{
  "scores": {
    "coverage_core_ruling": <float 0-4>,
    "conditions_exceptions": <float 0-2>,
    "factual_doctrinal_accuracy": <float 0-2>,
    "clarity_language": <float 0-1>,
    "directness_format": <float 0-1>
  },
  "overall": <float 0-10>,
  "critical_checks": {
    "contradicts_ground_truth": <true/false>,
    "omits_critical_conditions": <true/false>,
    "introduces_unlawful_elements": <true/false>,
    "hallucinated_citations": <true/false>,
    "non_answer_or_evasive": <true/false>,
    "off_topic_or_unsafe": <true/false>
  },
  "note": "<short NOTE in {NOTE_LANG}>"
}
""".strip()

RUBRIC_ISLAMIC = """
You are an expert evaluator in Islamic finance (Fiqh al-mu'āmalāt). You will be given, each time:
- topic (optional context) – may be empty
- question – in Arabic
- ground_truth – the reference correct answer, in Arabic
- candidate_answer – the model's answer to be evaluated, in Arabic

Your task:
- Judge how well the candidate_answer matches the ground_truth in *meaning*, *ruling*, *justification* and *constraints*.
- Prioritize doctrinal correctness and completeness of key conditions/exceptions.
- Do not penalize stylistic paraphrase if the core ruling and constraints are preserved.
- Be concise and deterministic.

Score on the following criteria so that the total sums to exactly 10:

1) Coverage of core ruling (0–4)
2) Conditions, exceptions, constraints (0–2)
3) Doctrinal/factual accuracy (0–2)
4) Clarity & Arabic language quality (0–1)
5) Directness & on-topic (0–1)

Critical checks (true/false):
- contradicts_ground_truth
- omits_critical_conditions
- introduces_unlawful_elements
- hallucinated_citations
- non_answer_or_evasive
- off_topic_or_unsafe

Output ONLY valid JSON with this schema:
{
  "scores": {
    "coverage_core_ruling": <float 0-4>,
    "conditions_exceptions": <float 0-2>,
    "factual_doctrinal_accuracy": <float 0-2>,
    "clarity_language": <float 0-1>,
    "directness_format": <float 0-1>
  },
  "overall": <float 0-10>,
  "critical_checks": {
    "contradicts_ground_truth": <true/false>,
    "omits_critical_conditions": <true/false>,
    "introduces_unlawful_elements": <true/false>,
    "hallucinated_citations": <true/false>,
    "non_answer_or_evasive": <true/false>,
    "off_topic_or_unsafe": <true/false>
  },
  "note": "<short NOTE in {NOTE_LANG}>"
}
""".strip()

# rubric_key (from tasks.yaml) -> (rubric, system_prompt, builder)
_SYS = {
    "financial_qa": "You are a precise, rigorous evaluator for Arabic financial QA. Score deterministically and output only valid JSON.",
    "fatwa_qa": "You are a precise, rigorous evaluator for Arabic fatwa QA. Output only valid JSON.",
    "islamic_qa": "You are a precise, rigorous evaluator for Islamic finance QA. Output only valid JSON.",
}


def _build_prompt(rubric_key, rec, note_lang="Arabic"):
    """Assemble the evaluation prompt EXACTLY as the original judge scripts do."""
    prompt = (rec.get("prompt") or "").strip()
    gt = (rec.get("reference") or "").strip()
    cand = (rec.get("generated_text") or "").strip()

    if rubric_key == "financial_qa":
        rubric = RUBRIC_FINANCIAL.replace("{NOTE_LANG}", note_lang)
        parts = [rubric, "\n---\n",
                 "The following fields are in ARABIC (do not translate them; evaluate against each other):",
                 "Note: 'prompt' already includes the financial report/excerpt and the question.",
                 f"prompt (AR):\n{prompt}",
                 f"ground_truth (AR):\n{gt}",
                 f"candidate_answer (AR):\n{cand}",
                 "\nReturn ONLY JSON now.\n"]
        return "\n".join(parts)

    if rubric_key == "fatwa_qa":
        rubric = RUBRIC_FATWA.replace("{NOTE_LANG}", note_lang)
        category = rec.get("category")
        cat_line = f"category (AR/label): {category}\n" if category else ""
        parts = [rubric, "\n---\n",
                 "The following fields are in ARABIC (do not translate them; evaluate against each other):",
                 cat_line,
                 f"prompt (AR):\n{prompt}",
                 f"ground_truth (AR):\n{gt}",
                 f"candidate_answer (AR):\n{cand}",
                 "\nReturn ONLY JSON now.\n"]
        return "\n".join([p for p in parts if p])

    if rubric_key == "islamic_qa":
        rubric = RUBRIC_ISLAMIC.replace("{NOTE_LANG}", note_lang)
        parts = [rubric, "\n---\n",
                 "The following fields are in ARABIC (do not translate them; evaluate against each other):",
                 f"question (AR):\n{prompt}",
                 f"ground_truth (AR):\n{gt}",
                 f"candidate_answer (AR):\n{cand}",
                 "\nReturn ONLY JSON now.\n"]
        return "\n".join(parts)

    raise ValueError(f"No judge rubric for '{rubric_key}'")


# ============================ JSON parsing (verbatim behaviour) =============

def _extract_json(text):
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in judge response.")
    return json.loads(m.group(0))


def _try_repair_json(raw):
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).replace("﻿", "")
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        return _extract_json(cleaned)
    except Exception:
        pass
    return json.loads(re.sub(r",\s*([}\]])", r"\1", cleaned))


# ============================ Judge =========================================

class Judge:
    def __init__(self, model="gpt-4o", base_url=None, api_key_env="OPENAI_API_KEY",
                 temperature=0.0, max_output_tokens=4096, note_language="Arabic",
                 max_workers=16, max_retries=5):
        key = os.getenv(api_key_env)
        if not key:
            raise RuntimeError(f"Judge API key env '{api_key_env}' is not set.")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.note_language = note_language
        self.max_workers = max_workers
        self.max_retries = max_retries

    def _score_one(self, rubric_key, rec):
        prompt = _build_prompt(rubric_key, rec, self.note_language)
        system = _SYS[rubric_key]
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": prompt}],
                )
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("Empty judge response.")
                try:
                    obj = json.loads(text)
                except Exception:
                    obj = _try_repair_json(text)
                overall = obj.get("overall")
                return {"id": rec.get("id"),
                        "judge_overall": float(overall) if overall is not None else None,
                        "judge_scores": obj.get("scores", {}),
                        "judge_critical_checks": obj.get("critical_checks", {}),
                        "judge_note": obj.get("note", "")}
            except Exception as e:
                last_err = e
                time.sleep(min(30, 2 ** attempt))
        return {"id": rec.get("id"), "judge_overall": None, "judge_note": f"ERROR: {last_err}"}

    def score_records(self, records, rubric_key):
        results = [None] * len(records)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self._score_one, rubric_key, r): i for i, r in enumerate(records)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        return results


def aggregate_scores(results):
    """Mean of judge_overall on the original 0-10 scale (matches the paper tables)."""
    vals = [r["judge_overall"] for r in results if r and r.get("judge_overall") is not None]
    n = len(vals)
    mean = sum(vals) / n if n else 0.0
    var = sum((v - mean) ** 2 for v in vals) / n if n else 0.0
    return {"mean": mean, "std": var ** 0.5, "n": n, "n_failed": len(results) - n}

"""
MCQ scoring — deterministic, no LLM judge.

The correct answer is a clean letter (dataset column `answer`, e.g. "d") which is
also stored as a 0-based index (`gold`, e.g. 3). We never need a model to know the
gold answer. We only need to read which letter the candidate model produced.

Two ways to read the model's choice (both implemented here):

  1. logprob ranking  -- the lm-eval-harness style. Given the first-token logprobs
     over the vocabulary, pick the candidate letter with the highest logprob.
     Deterministic, no parsing, no API. This is the primary signal.

  2. regex extraction -- parse the generated text for the answer letter. Used as a
     fallback when logprobs are unavailable (e.g. scoring API models or old runs).
"""

import re
import string

LETTERS = string.ascii_lowercase  # 'a','b','c',...


def index_to_letter(idx: int) -> str:
    return LETTERS[idx]


def gold_letter(item: dict, num_choices: int) -> str:
    """Return the gold answer as a lowercase letter, from `answer` or `gold` index."""
    ans = item.get("answer")
    if ans is not None:
        s = str(ans).strip().lower()
        if len(s) == 1 and s in LETTERS[:num_choices]:
            return s
    gold = item.get("gold")
    if gold is not None:
        s = str(gold).strip()
        if s.isdigit():
            i = int(s)
            if 0 <= i < num_choices:
                return index_to_letter(i)
        elif len(s) == 1 and s.lower() in LETTERS[:num_choices]:
            return s.lower()
    return ""  # unknown


# --- regex extraction of the predicted letter from free text -----------------

# Arabic + English answer cues, ordered from most to least specific.
_PATTERNS = [
    r"الإجابة\s*(?:الصحيحة)?\s*(?:هي)?\s*[:\-]?\s*\(?\**\s*([a-z])\b",
    r"الجواب\s*(?:الصحيح)?\s*(?:هو)?\s*[:\-]?\s*\(?\**\s*([a-z])\b",
    r"\banswer\b\s*(?:is)?\s*[:\-]?\s*\(?\**\s*([a-z])\b",
    r"^\s*\(?\**\s*([a-z])\s*[\.\):\-]",   # leading "b)" / "b." / "b:"
    r"\(\s*([a-z])\s*\)",                   # "(b)"
    r"(?:^|\s)([a-z])\s*$",                 # a bare trailing letter
]


def extract_letter(text: str, num_choices: int) -> str:
    """Best-effort parse of the chosen letter. Returns '' if not found."""
    if not text:
        return ""
    valid = set(LETTERS[:num_choices])
    low = text.strip().lower()

    # Fast path: the whole answer is just the letter.
    if len(low) == 1 and low in valid:
        return low

    for pat in _PATTERNS:
        m = re.search(pat, low, flags=re.MULTILINE)
        if m and m.group(1) in valid:
            return m.group(1)

    # Last resort: first valid standalone letter token anywhere.
    for tok in re.findall(r"\b([a-z])\b", low):
        if tok in valid:
            return tok
    return ""


def letter_from_logprobs(first_token_logprobs: dict, num_choices: int) -> str:
    """
    Pick the candidate letter with the highest logprob.

    `first_token_logprobs` maps a decoded token string -> logprob (float), as
    produced for the first generated position. Tokens are normalised (stripped,
    lowercased) so " B" and "b" both count for letter 'b'.
    """
    valid = set(LETTERS[:num_choices])
    best, best_lp = "", float("-inf")
    for tok, lp in first_token_logprobs.items():
        t = str(tok).strip().lower()
        if t in valid and lp > best_lp:
            best, best_lp = t, lp
    return best


def score_record(item: dict, num_choices: int,
                 first_token_logprobs: dict = None) -> dict:
    """
    Produce a per-example scoring record.

    Prediction priority: logprob ranking (if logprobs given) -> regex extraction.
    """
    gold = gold_letter(item, num_choices)
    gen = item.get("generated_text", "") or ""

    pred_lp = letter_from_logprobs(first_token_logprobs, num_choices) if first_token_logprobs else ""
    pred_rx = extract_letter(gen, num_choices)
    pred = pred_lp or pred_rx

    return {
        "id": item.get("id"),
        "gold": gold,
        "pred": pred,
        "pred_logprob": pred_lp,
        "pred_regex": pred_rx,
        "method": "logprob" if pred_lp else ("regex" if pred_rx else "none"),
        "correct": bool(pred) and pred == gold,
        "invalid": pred == "",
    }


def accuracy(records: list) -> dict:
    """Aggregate per-example records into accuracy stats."""
    total = len(records)
    correct = sum(r["correct"] for r in records)
    invalid = sum(r["invalid"] for r in records)
    return {
        "accuracy": 100.0 * correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "invalid": invalid,
        "invalid_rate": 100.0 * invalid / total if total else 0.0,
    }

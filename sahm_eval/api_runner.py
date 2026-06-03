"""
API backend for closed/hosted models (OpenAI-compatible chat endpoints).

Mirrors VLLMModel's interface so run.py is backend-agnostic. Chat APIs don't
expose token logprobs reliably, so MCQ predictions here are read from the
generated text via regex (sahm_eval.mcq.extract_letter).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from .prompts import MCQ_INSTRUCTION


class APIModel:
    def __init__(self, cfg: dict, max_workers: int = 16, max_retries: int = 4):
        key = os.getenv(cfg.get("api_key_env", "OPENAI_API_KEY"))
        if not key:
            raise RuntimeError(f"API key env '{cfg.get('api_key_env')}' is not set.")
        self.client = OpenAI(api_key=key, base_url=cfg.get("base_url"))
        self.model_name = cfg["model_name"]
        self.max_tokens = cfg.get("max_tokens", 1024)
        self.max_workers = max_workers
        self.max_retries = max_retries

    def _chat(self, user_text: str, max_tokens: int) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name, temperature=0.0, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": user_text}],
                )
                return resp.choices[0].message.content or ""
            except Exception:
                if attempt == self.max_retries - 1:
                    return ""
                time.sleep(2 ** attempt)

    def _map(self, texts, max_tokens):
        out = [""] * len(texts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self._chat, t, max_tokens): i for i, t in enumerate(texts)}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
        return out

    def run_mcq(self, records, max_new_tokens=8, seed=42):
        texts = [f"{MCQ_INSTRUCTION}\n\n{r['prompt']}" for r in records]
        gens = self._map(texts, max_new_tokens)
        return [{**r, "generated_text": g, "first_token_logprobs": {}}
                for r, g in zip(records, gens)]

    def run_generative(self, records, max_new_tokens=1024, seed=42, batch_size=64):
        texts = [r["prompt"] for r in records]
        gens = self._map(texts, max_new_tokens)
        return [{**r, "generated_text": g} for r, g in zip(records, gens)]

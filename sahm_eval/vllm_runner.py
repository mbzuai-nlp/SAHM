"""
vLLM inference backend (runs on a GPU node).

- MCQ:        single forward, max_tokens small, logprobs captured. We keep both the
              generated text AND the first-token logprobs so scoring can use the
              lm-harness-style logprob ranking with a regex fallback.
- generative: standard batched generation.

Prompts whose tokens + max_new_tokens would exceed the model's context window are
LEFT-truncated on the raw content (keeping the question/instruction at the end and
re-applying the chat template), exactly as lm-evaluation-harness does. Truncation
is never silent — the number of truncated examples is printed.
"""

import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from .prompts import MCQ_INSTRUCTION, STOP


class VLLMModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_id = cfg["model_id"]
        self.use_chat_template = cfg.get("use_chat_template", True)
        self.max_model_len = cfg.get("max_model_len", 4096)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=os.getenv("HF_TOKEN"))
        self.llm = LLM(
            model=self.model_id,
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
            max_num_seqs=cfg.get("max_num_seqs", 256),
            trust_remote_code=cfg.get("trust_remote_code", False),
        )

    def _apply_template(self, user_text: str) -> str:
        if self.use_chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False, add_generation_prompt=True)
        return f"{user_text}\n\n"

    def _build_prompts(self, user_texts, max_new_tokens):
        """Format prompts, left-truncating raw content that would overflow the window.

        Reserves room for generation plus a margin for chat-template tokens, so the
        final templated prompt always fits ``max_model_len``.
        """
        budget = self.max_model_len - max_new_tokens - 16
        prompts, n_trunc = [], 0
        for text in user_texts:
            ids = self.tokenizer(text, add_special_tokens=False).input_ids
            if len(ids) > budget:
                text = self.tokenizer.decode(ids[-budget:])  # keep the tail (the question)
                n_trunc += 1
            prompts.append(self._apply_template(text))
        if n_trunc:
            print(f"     ⚠️  left-truncated {n_trunc}/{len(user_texts)} prompts to fit "
                  f"the {self.max_model_len}-token context window")
        return prompts

    def run_mcq(self, records: list, max_new_tokens: int = 8, seed: int = 42) -> list:
        user_texts = [f"{MCQ_INSTRUCTION}\n\n{r['prompt']}" for r in records]
        prompts = self._build_prompts(user_texts, max_new_tokens)
        sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0,
                            seed=seed, logprobs=20, stop=STOP)
        outs = self.llm.generate(prompts, sp)
        results = []
        for r, out in zip(records, outs):
            o = out.outputs[0]
            ftlp = {}
            if o.logprobs:
                for _tok_id, lp in o.logprobs[0].items():
                    ftlp[lp.decoded_token] = lp.logprob
            results.append({**r, "generated_text": o.text.strip(),
                            "first_token_logprobs": ftlp})
        return results

    def run_generative(self, records: list, max_new_tokens: int = 1024,
                       seed: int = 42, batch_size: int = 64) -> list:
        prompts = self._build_prompts([r["prompt"] for r in records], max_new_tokens)
        sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, seed=seed, stop=STOP)
        outs = self.llm.generate(prompts, sp)
        return [{**r, "generated_text": out.outputs[0].text.strip()}
                for r, out in zip(records, outs)]

# SAHM: A Benchmark for Arabic Financial and Shari'ah-Compliant Reasoning

<div align="left" style="margin:24px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif"
       width="100%" height="4"/>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2604.19098"><img src="https://img.shields.io/badge/arXiv-Paper-brightgreen?style=flat-square" alt="arXiv"></a>
  <a href="https://huggingface.co/SahmBenchmark"><img src="https://img.shields.io/badge/🤗-Datasets-yellow?style=flat-square" alt="Hugging Face"></a>
  <a href="https://github.com/mbzuai-nlp/SAHM/stargazers"><img src="https://img.shields.io/github/stars/mbzuai-nlp/SAHM?style=flat-square" alt="GitHub Repo stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg?style=flat-square" alt="License"></a>
</p>

<p align="center">
  <a href="https://scholar.google.com/citations?user=ic1jai8AAAAJ&hl=en"><b>Rania Elbadry</b></a>,
  <b>Sarfraz Ahmad</b>, <a href="https://ahmedheakl.github.io/"><b>Ahmed Heakl</b></a>,
  <b>Dani Bouch</b>, <b>Momina Ahsan</b>, <b>Muhra AlMahri</b>, <b>Marwa Elsaid Khalil</b>,<br>
  <b>Yuxia Wang</b>, <b>Salem Lahlou</b>, <b>Sophia Ananiadou</b>, <b>Veselin Stoyanov</b>,
  <b>Jimin Huang</b>, <b>Xueqing Peng</b>, <b>Preslav Nakov</b>, <b>Zhuohan Xie</b>
</p>

<p align="center"><b>MBZUAI</b></p>

<!-- Optional banner. Drop the image at assets/sahm-overview.png and it will render here. -->
<p align="center">
  <img src="assets/sahm-overview.png" alt="SAHM benchmark overview" width="720"/>
</p>

---

## 🆕 Latest Updates
- 📢 **June 2026**: Evaluation harness released.

## Table of Contents
- [💡 TL;DR](#-tldr)
- [📚 The Benchmark](#-the-benchmark)
- [🧠 How Scoring Works](#-how-scoring-works)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📊 Output & Leaderboard](#-output--leaderboard)
- [🔁 Reproducibility](#-reproducibility)
- [🧩 Adding Models & Tasks](#-adding-models--tasks)
- [🗂️ Project Layout](#️-project-layout)
- [📚 Citation](#-citation)

## 💡 TL;DR

**SAHM** is the first Arabic benchmark for financial and Shari'ah-compliant
reasoning — **14,380 expert-validated examples** across multiple-choice and
generative tasks spanning accounting, business, financial sentiment, event-cause
reasoning, fatwa, Islamic-finance Q&A, and extractive summarization.

This repository is the **official evaluation harness**. One command evaluates any
model across all tasks for multiple seeds and emits a `mean ± std` leaderboard:

```bash
sahm-eval run --model qwen2.5-7b --tasks all --runs 3
```

- **Deterministic MCQ scoring** — answers are read by first-token **log-probability
  ranking** (à la `lm-evaluation-harness`), not by a second LLM. No API cost, no
  parsing ambiguity, fully reproducible.
- **Faithful generative judging** — reference-based LLM-as-judge using the **exact
  published rubrics**; swap between GPT and Gemini with one flag.
- **One config-driven CLI** — tasks and models declared in YAML; secrets read from
  the environment only.

> **Key finding from the paper:** Arabic fluency does not imply financial
> reasoning — models that score ~91% on recognition tasks degrade sharply on
> generation, with event-cause reasoning showing the widest spread across 20 LLMs.

## 📚 The Benchmark

| Task | Key | Type | Metric | Judge |
|---|---|---|---|---|
| Accounting MCQ | `accounting` | MCQ | accuracy (%) | — |
| Business MCQ | `business` | MCQ | accuracy (%) | — |
| Islamic Fatwa MCQ | `fatwa_mcq` | MCQ | accuracy (%) | — |
| Financial Sentiment MCQ | `sentiment` | MCQ | accuracy (%) | — |
| Financial QA | `financial_qa` | Generative | judge `overall` 0–10 | ✅ |
| Fatwa QA | `fatwa_qa` | Generative | judge `overall` 0–10 | ✅ |
| Islamic Finance QA | `islamic_qa` | Generative | judge `overall` 0–10 | ✅ |
| Financial Summarization | `summarization` | Generative | generation only | — |

All datasets live on the Hub under [🤗 SahmBenchmark](https://huggingface.co/SahmBenchmark)
and are loaded automatically (set `HF_TOKEN`).

## 🧠 How Scoring Works

<!-- Optional diagram. Drop the image at assets/sahm-pipeline.png and it will render here. -->
<p align="center">
  <img src="assets/sahm-pipeline.png" alt="SAHM evaluation pipeline: load → generate → score → leaderboard" width="760"/>
</p>

**MCQ — no LLM in the loop.** The gold answer is a clean letter (`answer`, e.g.
`d`) and index (`gold`, e.g. `3`). For each question we read the model's chosen
letter from its first-token log-probabilities and take the arg-max over the valid
options (`a`–`e`), with a robust regex fallback. This is deterministic, free, and
reproducible — the standard `lm-evaluation-harness` approach for multiple choice.

**Generative — reference-based judge.** For Financial / Fatwa / Islamic-Finance QA,
an LLM judge scores each answer 1–10 against the reference using the project's
original, task-specific rubrics (correctness, conditions/exceptions, faithfulness,
critical checks). Summarization is generated but not auto-scored (no rubric in the
original pipeline).

**Long prompts** are left-truncated to the model's context window (keeping the
question and re-applying the chat template), with a printed count — never silently.

## 📦 Installation

```bash
git clone https://github.com/mbzuai-nlp/SAHM.git
cd SAHM

pip install -e .            # core: data loading, API models, judge
pip install -e ".[gpu]"     # add local inference with vLLM (GPU node)
```

Set credentials once (read from the environment, never hardcoded):

```bash
cp .env.example .env        # fill in HF_TOKEN, OPENAI_API_KEY
set -a; source .env; set +a
```

## 🚀 Quick Start

```bash
# list available tasks and models
sahm-eval list

# fast smoke test (10 examples, no judge)
sahm-eval run --model qwen2.5-7b --tasks accounting --limit 10 --skip-judge

# full evaluation: all tasks, 3 seeds, on a GPU node
sahm-eval run --model qwen2.5-7b --tasks all --runs 3
```

Evaluate a **hosted API model** (no GPU; generation via API, MCQ read by regex):

```bash
sahm-eval run --model gpt-4o --backend api --tasks all --runs 3
```

Choose the **judge** for generative tasks:

```bash
# GPT-4o judge (default; needs OPENAI_API_KEY)
sahm-eval run --model qwen2.5-7b --tasks financial_qa --judge-model gpt-4o

# Gemini judge via its OpenAI-compatible endpoint (needs GEMINI_API_KEY)
sahm-eval run --model qwen2.5-7b --tasks financial_qa \
  --judge-model gemini-2.5-flash \
  --judge-base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --judge-key-env GEMINI_API_KEY
```

Re-score MCQ generations you already have — **offline, no GPU, no judge**:

```bash
sahm-eval score-mcq --tree results/                 # a whole tree, mean ± std
sahm-eval score-mcq path/to/generations.jsonl       # a single file
```

## 📊 Output & Leaderboard

```
results/run_<timestamp>_<model>/
├── run_1_seed42/<Task>/generations.jsonl   # raw outputs + per-example scores
├── run_1_seed42/<Task>/score.json          # this seed's score
├── run_1_seed42/<Task>/judge.jsonl         # judge verdicts (generative)
├── leaderboard.md                          # MCQ % and judge /10 tables
└── leaderboard.json
```

`leaderboard.md` reports MCQ accuracy and the generative judge score in separate
tables (the two scales are never mixed):

```
## MCQ — accuracy (%)
| Rank | Model | Accounting | Business | ... | Avg (%) |
| 1 | qwen2.5-7b | 41.0±1.0 | 72.3±0.8 | ... | ... |

## Generative — LLM-judge overall (0–10)
| Rank | Model | Financial QA | Fatwa QA | Islamic QA | Avg (/10) |
| 1 | qwen2.5-7b | 6.2±0.2 | ... | ... | ... |
```

## 🔁 Reproducibility

The prompts that produced the published numbers are reused **verbatim** — do not
edit them:

- **MCQ instruction** — `sahm_eval/prompts.py`, identical to the original.
- **QA prompt** — the dataset's own `prompt` field, no wrapper added.
- **Judge rubrics / system prompts / JSON schema** — `sahm_eval/judge.py`, copied
  verbatim from the original judge scripts. The rubric is backend-agnostic, so
  `--judge-model gpt-4o` reproduces the GPT judge and `--judge-model gemini-2.5-flash`
  the Gemini judge.

Runs are deterministic (greedy decoding, fixed seeds). The single intentional
change from the original pipeline is the **MCQ scoring method** (log-prob/regex
instead of an LLM extraction call) — it changes scoring, not prompts.

## 🧩 Adding Models & Tasks

- **Model** — add a block to `sahm_eval/configs/models.yaml` (`open_models` for
  local vLLM, `api_models` for hosted OpenAI-compatible endpoints).
- **Task** — add a block to `sahm_eval/configs/tasks.yaml` with its Hub path,
  `type`, columns, and either `num_choices` (MCQ) or a `rubric` key (generative).
- Or point the CLI at your own files: `--tasks-file ... --models-file ...`.

## 🗂️ Project Layout

```
sahm_eval/
├── cli.py            sahm-eval entrypoint (run / score-mcq / list)
├── pipeline.py       orchestration: load → generate → score → leaderboard
├── loader.py         load & normalise a task from the Hub
├── vllm_runner.py    local generation (+ MCQ first-token logprobs)
├── api_runner.py     hosted OpenAI-compatible models
├── mcq.py            deterministic MCQ scoring (logprob + regex)
├── judge.py          reference-based LLM-as-judge (verbatim rubrics)
├── aggregate.py      mean ± std leaderboard (MCQ % / judge /10, unmixed)
├── rescore.py        offline re-scoring of existing generations
└── configs/          tasks.yaml, models.yaml (packaged defaults)
```

## 📚 Citation

```bibtex
@article{elbadry2026sahm,
  title   = {SAHM: A Benchmark for Arabic Financial and Shari'ah-Compliant Reasoning},
  author  = {Elbadry, Rania and Ahmad, Sarfraz and Heakl, Ahmed and Bouch, Dani and
             Ahsan, Momina and AlMahri, Muhra and Khalil, Marwa Elsaid and Wang, Yuxia and
             Lahlou, Salem and Ananiadou, Sophia and Stoyanov, Veselin and Huang, Jimin and
             Peng, Xueqing and Nakov, Preslav and Xie, Zhuohan},
  journal = {arXiv preprint arXiv:2604.19098},
  year    = {2026}
}
```

## License

Released under the [Apache License 2.0](LICENSE).

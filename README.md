<p align="center">
  <img src="docs/assets/logo.png" alt="SAHM Logo" width="200"/>
</p>

<h1 align="center" style="font-size: 42px; font-weight: bold; margin-top: -10px;">
  SAHM
</h1>

<p align="center" style="font-size: 20px; font-style: italic; margin-top: -10px;">
  Arabic Financial Instruction-Tuning Dataset & Models
</p>

<hr style="width: 80%; margin: 30px auto;">

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/paper-ArXiv-red?style=for-the-badge"></a>
  <a href="https://mbzuai-nlp.github.io/SAHM/"><img src="https://img.shields.io/badge/project-Page-brightgreen?style=for-the-badge"></a>
  <a href="https://mbzuai-nlp.github.io/SAHM/leaderboard.html"><img src="https://img.shields.io/badge/leaderboard-Page-blue?style=for-the-badge"></a>
  <a href="#"><img src="https://img.shields.io/badge/demo-Spaces-gold?style=for-the-badge"></a>
</p>

---

## 🌟 Overview

**SAHM** is the first **large-scale Arabic financial NLP benchmark** covering both **modern financial analysis** and **Islamic/Shari’ah-compliant reasoning**, introduced in our paper _SAHM: Arabic Financial Instruction-Tuning Dataset And Models_.

It includes **14,000+ high-quality Arabic samples** across **eight tasks**, derived from:

- AAOIFI Shari’ah standards
- Official fatwa archives
- Corporate earnings reports
- Market news
- Business and accounting exams
- Islamic finance regulatory material

SAHM also introduces:  
🟦 **SAHM-7B-Instruct**, an Arabic financial instruction-tuned model  
🟦 A unified evaluation framework  
🟦 First-of-its-kind datasets for Islamic finance + Arabic corporate analysis

---

## 📌 Features

Our benchmark includes eight diverse tasks:

1. **Islamic Finance Shari’ah Standards QA**
2. **Islamic Financial Fatwa QA**
3. **Islamic Financial Fatwa MCQ**
4. **Business MCQ**
5. **Accounting MCQ**
6. **Financial Report Sentiment Analysis**
7. **Report Extractive Summarization**
8. **Event–Cause Reasoning QA**

These tasks reflect **real Arabic financial workflows**, combining modern finance with Islamic jurisprudence (fiqh al-muʿāmalāt).

---

## 📁 Dataset Structure

Each dataset adheres to a unified JSON schema and standardized evaluation protocol.

| Task                     | #Train | #Eval | Format  | Capability                      |
| ------------------------ | ------ | ----- | ------- | ------------------------------- |
| Shari’ah Standards QA    | 1621   | 406   | QA      | Islamic finance legal reasoning |
| Islamic Fatwa QA         | 11,703 | 250   | QA      | Faith-based financial rulings   |
| Event–Cause Reasoning    | 160    | 40    | QA      | Financial causal inference      |
| Extractive Summarization | 160    | 40    | Summary | Financial disclosure extraction |
| Fatwa MCQ                | –      | 250   | MCQ     | Recognition-style reasoning     |
| Business MCQ             | 381    | 76    | MCQ     | Business fundamentals           |
| Accounting MCQ           | 95     | 24    | MCQ     | Numerical & IFRS reasoning      |
| Sentiment Analysis       | 160    | 40    | MCQ     | Financial polarity detection    |

Full details appear in Table 1 of the paper.

---

## 🧠 SAHM-7B-Instruct Model

A **7B Arabic-centric model** instruction-tuned on all SAHM datasets.  
Built on top of **ALLAM-7B**, it achieves:

- **Best MCQ performance among Arabic/open models**
- **+37.5 improvement in Accounting MCQ**
- **Strong business & sentiment accuracy**
- Competent but still developing **open-ended reasoning**

See Table 2 in the paper for full comparison.

---

## 🏆 Leaderboard (MCQ Tasks)

| Model                       | Mean Accuracy (%) |
| --------------------------- | ----------------- |
| **GPT-5**                   | **73.9**          |
| GPT-4o                      | 67.0              |
| Qwen2.5-72B                 | 60.4              |
| Fanar-1-9B                  | 53.9              |
| ALLAM-7B                    | 56.1              |
| **SAHM-7B-Instruct (ours)** | **71.7**          |

---

## 🏆 Leaderboard (Open-Ended QA)

Average Judge Score (0–10):

| Model                | Score    |
| -------------------- | -------- |
| **GPT-5**            | **8.98** |
| Claude 4 Sonnet      | 7.77     |
| GPT-4o               | 7.08     |
| Gemini 2.5 Pro       | 5.73     |
| ALLAM-7B             | 5.05     |
| Fanar-1-9B           | 4.82     |
| **SAHM-7B-Instruct** | **5.07** |

(Open-ended tasks remain significantly harder for current Arabic models.)

---

## 📝 Installation

```bash
git clone https://github.com/mbzuai-nlp/SAHM
cd SAHM
pip install -r requirements.txt
```

## 💾 Using the Dataset

```python
from sahm import load_dataset

ds = load_dataset("sahm", "fatwa_qa")
print(ds["train"][0])
```

## 🤖 Using SAHM-7B-Instruct

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("mbzuai-nlp/SAHM-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("mbzuai-nlp/SAHM-7B-Instruct")

prompt = "اشرح حكم بيع المرابحة في حالة عدم تملك السلعة."
out = model.generate(**tok(prompt, return_tensors="pt"))
print(tok.decode(out[0], skip_special_tokens=True))
```

## 🗂 Repository Structure

```bash
SAHM/
│
├── data/
│   ├── shariah_standards/
│   ├── fatwa_qa/
│   ├── mcq/
│   ├── sentiment/
│   ├── summarization/
│   └── event_cause/
│
├── models/
│   └── SAHM-7B-Instruct/
│
├── docs/
│   └── assets/logo.png
│
├── evaluation/
└── README.md
```

## 📘 Citation

If you use SAHM, please cite:

```bibtex
@article{sahm2025,
  title={SAHM: Arabic Financial Instruction-Tuning Dataset And Models},
  author={Elbadry, Rania and Ahmad, Sarfraz and Bouch, Dani and Ahsan, Momina and Peng, Xueqing and Huang, Jimin and AlMahri, Muhra and Khalil, Marwa Elsaid and Wang, Yuxia and Lahlou, Salem and Stoyanov, Veselin and Ananiadou, Sophia and Nakov, Preslav and Xie, Zhuohan},
  year={2025},
  institution={MBZUAI}
}
```

## 🏷 License

The dataset and code are released under MIT License

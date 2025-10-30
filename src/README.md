# Multi-Dataset Arabic Evaluation Framework

This repository contains tools for zero-shot evaluation of Arabic datasets including Multiple Choice Questions (MCQ) and Question-Answering (QA) tasks using both open-source and closed models.

## Features

- **Multi-dataset support**: Evaluate 8 different Arabic datasets (4 MCQ + 4 QA)
- **Zero-shot evaluation**: Direct evaluation without fine-tuning or few-shot examples
- **Dual model support**: Both open-source (local) and closed (API) models
- **Organized output**: Results organized by dataset type and name in separate directories
- **Resume capability**: Safe resume functionality for interrupted runs
- **Flexible filtering**: Filter by models, splits, or datasets

## Datasets

The evaluation supports the following datasets:

### MCQ Datasets

1. **arabic-accounting-mcq_eval** - `SahmBenchmark/arabic-accounting-mcq_eval`
2. **arabic-business-mcq_eval** - `SahmBenchmark/arabic-business-mcq_eval`
3. **fatwa-mcq-evaluation_standardized** - `SahmBenchmark/fatwa-mcq-evaluation_standardized`
4. **Sentiment_Analysis_MCQ_eval** - `SahmBenchmark/Sentiment_Analysis_MCQ_eval`

### QA Datasets

1. **arabic-financial-qa_eval** - `SahmBenchmark/arabic-financial-qa_eval`
2. **fatwa-qa-evaluation** - `SahmBenchmark/fatwa-qa-evaluation`
3. **financial-reports-extractive-summarization_eval** - `SahmBenchmark/financial-reports-extractive-summarization_eval`
4. **Islamic_Finance_QnA_eval** - `SahmBenchmark/Islamic_Finance_QnA_eval`

## Models

### Open Source Models (Local Inference)

- `gemma-2-9b-it` - Google Gemma 2 9B Instruct
- `gemma-3-12b-it` - Google Gemma 3 12B Instruct
- `llama-3.1-8b` - Meta Llama 3.1 8B Instruct
- `qwen2.5-7b-instruct` - Qwen 2.5 7B Instruct
- `qwen2.5-14b-instruct` - Qwen 2.5 14B Instruct

### Closed Models

**Via DeepInfra API:**

- `claude-4-sonnet` - Anthropic Claude 4 Sonnet
- `gemini-2.5-pro` - Google Gemini 2.5 Pro (via Bloomberg GPT-50B)

**Via OpenAI API:**

- `gpt-4` - OpenAI GPT-4
- `gpt-4o` - OpenAI GPT-4o
- `gpt-5` - OpenAI GPT-5

## Setup

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys** (for closed models):
   Create a `.env` file in the project root:

   ```bash
   # For DeepInfra models (Claude, Gemini)
   DEEPINFRA_API_KEY=your_deepinfra_api_key_here

   # For OpenAI models (GPT-4, GPT-4o, GPT-5)
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Configure HuggingFace Access** (for dataset access):
   Add your HuggingFace access token to the `.env` file:

   ```bash
   HF_TOKEN=your_huggingface_token_here
   ```

   Then authenticate with HuggingFace CLI:

   ```bash
   huggingface-cli login
   ```

   This enables access to the SahmBenchmark organization datasets with read-only permissions.

## Usage

### Run All Datasets (Recommended)

Run zero-shot evaluation on all 8 datasets (4 MCQ + 4 QA) with all models:

```bash
# Using the convenience script
./run_all.sh

# Or directly
python scripts/eval.py \
    --config config.yaml \
    --models models.yaml \
    --model-type all
```

### Filtered Evaluation

Run specific combinations for zero-shot evaluation:

```bash
# Only specific datasets (MCQ and QA)
python scripts/eval.py \
    --config config.yaml \
    --models models.yaml \
    --model-type all \
    --dataset-filter "arabic-accounting-mcq_eval,arabic-financial-qa_eval"

# Only MCQ datasets
python scripts/eval.py \
    --config config.yaml \
    --models models.yaml \
    --model-type all \
    --dataset-filter "arabic-accounting-mcq_eval,arabic-business-mcq_eval,fatwa-mcq-evaluation_standardized,Sentiment_Analysis_MCQ_eval"

# Only QA datasets
python scripts/eval.py \
    --config config.yaml \
    --models models.yaml \
    --model-type all \
    --dataset-filter "arabic-financial-qa_eval,fatwa-qa-evaluation,financial-reports-extractive-summarization_eval,Islamic_Finance_QnA_eval"

# Only specific models
python scripts/eval.py \
    --config config.yaml \
    --models models.yaml \
    --model-type all \
    --model-filter "claude-4-sonnet,gemma-2-9b-it"

# Only specific splits
python scripts/eval.py \
    --config config.yaml \
    --models models.yaml \
    --model-type all \
    --split-filter "validation"

# Only open source models
python scripts/eval.py \
    --config config.yaml \
    --models models.yaml \
    --model-type open

# Only closed models
python scripts/eval.py \
    --config config.yaml \
    --models models.yaml \
    --model-type closed

# Single dataset example (MCQ)
python scripts/eval.py \
    --config config.yaml \
    --models models.yaml \
    --model-type all \
    --dataset-filter "arabic-accounting-mcq_eval"

# Single dataset example (QA)
python scripts/eval.py \
    --config config.yaml \
    --models models.yaml \
    --model-type all \
    --dataset-filter "arabic-financial-qa_eval"
```

### Test OpenAI API Integration

To test that your OpenAI API key is working correctly, run a quick test on one dataset:

```bash
# Test GPT-4o on arabic-business-mcq_eval (validation split only)
python scripts/eval.py \
    --config config.yaml \
    --models models.yaml \
    --model-type closed \
    --dataset-filter "arabic-business-mcq_eval" \
    --model-filter "gpt-4o" \
    --split-filter "validation"

# Test GPT-4o on arabic-financial-qa_eval (validation split only)
python scripts/eval.py \
    --config config.yaml \
    --models models.yaml \
    --model-type closed \
    --dataset-filter "arabic-financial-qa_eval" \
    --model-filter "gpt-4o" \
    --split-filter "validation"
```

Alternative: Run from scripts directory:

```bash
cd scripts
python eval.py \
    --config ../config.yaml \
    --models ../models.yaml \
    --model-type closed \
    --dataset-filter "arabic-business-mcq_eval" \
    --model-filter "gpt-4o" \
    --split-filter "validation"
```

# Custom combination example (MCQ and QA datasets)

python scripts/eval.py \
 --config config.yaml \
 --models models.yaml \
 --model-type closed \
 --dataset-filter "arabic-business-mcq_eval,fatwa-qa-evaluation" \
 --split-filter "validation" \
 --model-filter "claude-4-sonnet,gpt-4o"

```

## Output Structure

Results are organized by dataset type (mcq/qa) and then by dataset name:

```

outputs/
├── mcq/
│ ├── arabic-accounting-mcq_eval/
│ │ ├── validation_claude-4-sonnet_results.jsonl
│ │ ├── validation_gemma-2-9b-it_results.jsonl
│ │ ├── test_claude-4-sonnet_results.jsonl
│ │ ├── test_gemma-2-9b-it_results.jsonl
│ │ └── ...
│ ├── arabic-business-mcq_eval/
│ │ ├── validation_claude-4-sonnet_results.jsonl
│ │ └── ...
│ ├── fatwa-mcq-evaluation_standardized/
│ │ └── ...
│ └── Sentiment_Analysis_MCQ_eval/
│ └── ...
└── qa/
├── arabic-financial-qa_eval/
│ ├── validation_claude-4-sonnet_results.jsonl
│ ├── test_gemma-2-9b-it_results.jsonl
│ └── ...
├── fatwa-qa-evaluation/
│ └── ...
├── financial-reports-extractive-summarization_eval/
│ └── ...
└── Islamic_Finance_QnA_eval/
└── ...

````

## Result Format

### MCQ Results
Each MCQ result file contains JSONL format with the original dataset fields plus:

```json
{
  "id": "item_001",
  "query": "Original question text",
  "choices": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
  "answer": "A",
  "model_response": "Generated model response"
}
````

### QA Results

Each QA result file contains JSONL format with the original dataset fields plus:

```json
{
  "id": "qa_001",
  "question": "Original question text",
  "answer": "Reference answer",
  "model_response": "Generated model response"
}
```

## Performance Estimates

**Total Evaluations**: 160 model-split combinations

- 8 datasets (4 MCQ + 4 QA) × 10 models × 2 splits = 160 evaluations
- Estimated time: 4-8 hours (depending on hardware and API limits)
- Note: QA tasks typically take longer than MCQ due to longer generation sequences

## Configuration Files

- `config.yaml` - Main configuration file (datasets, generation params, API settings)
- `models.yaml` - Model definitions and selection

## Zero-shot Evaluation Approach

This framework performs zero-shot evaluation, meaning:

- **No fine-tuning**: Models are used as-is without training on task-specific data
- **No few-shot examples**: No example questions/answers are provided to the model
- **Direct evaluation**: Models generate responses based solely on the task instructions and input

## Resume Functionality

The evaluation supports safe resume. If interrupted, simply re-run the same command and it will continue from where it left off by checking existing result files.

## Troubleshooting

1. **CUDA/GPU Issues**: The script automatically detects available hardware and adjusts accordingly
2. **API Rate Limits**: Built-in exponential backoff and retry logic
3. **Memory Issues**: Consider running with `--model-type closed` to use only API models
4. **Import Errors**: If you get `ModuleNotFoundError`, try running from the scripts directory:
   ```bash
   cd scripts
   python eval.py --config ../config.yaml --models ../models.yaml [other options]
   ```
5. **API Key Issues**:
   - Ensure your `.env` file is in the project root
   - Verify API keys are correctly formatted
   - Test with a simple model first (e.g., GPT-4o on validation split only)
6. **Dataset Access**: Ensure you have access to the HuggingFace datasets:
   - Set up your HuggingFace token in `.env` file
   - Authenticate using `huggingface-cli login`
   - Verify access to SahmBenchmark organization datasets

## Scripts

- `scripts/eval.py` - Multi-dataset evaluation script for both MCQ and QA (main script)
- `scripts/utils.py` - Utility functions
- `scripts/api_client.py` - API client for closed models
- `run_all.sh` - Convenience script to run all datasets

## Additional Evaluation Scripts

- `evaluation/eval_mcq.py` - Specialized MCQ evaluation metrics
- `evaluation/run_eval_arabic_financial.py` - Arabic financial QA evaluation
- `evaluation/run_eval_fatwa.py` - Fatwa QA evaluation
- `evaluation/run_eval_islamic_financial.py` - Islamic finance QA evaluation

#!/usr/bin/env python3
"""
Arabic Models MCQ Evaluation Script - Cleaned & Hardened

Key changes vs your version:
- Adds set_env(), cleanup_memory(), clear_disk_cache() and runs them BEFORE EACH MODEL.
- Redirects HF/Torch caches to /scratch to avoid home quota issues.
- Fixes CUDA device mismatch by NOT moving inputs to a single device when using sharded models.
- Optional single-GPU mode by default (change visible_devices in set_env()).
- No hardcoded HF token; read from env (HF_TOKEN) if required.
"""

import os
import json
import time
import gc
import shutil
import glob
import warnings
from datetime import datetime

import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns  # you imported it; keeping it though unused
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


warnings.filterwarnings('ignore')



# Read HF token from environment if needed (avoid hardcoding)
HF_TOKEN = os.getenv("HF_TOKEN", None)
HF_TOKEN = "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"  # <-- your token here

# Confirmed accessible datasets
MCQ_DATASETS = [
    "SahmBenchmark/arabic-accounting-mcq_eval",
    "SahmBenchmark/arabic-business-mcq_eval",
    "SahmBenchmark/fatwa-mcq-evaluation_standardized",
    "SahmBenchmark/Sentiment_Analysis_MCQ_eval"
]

# Dataset display names for tables
DATASET_DISPLAY_NAMES = {
    "arabic-accounting-mcq_eval": "Accounting",
    "arabic-business-mcq_eval": "Business",
    "fatwa-mcq-evaluation_standardized": "Islamic Fatwa",
    "Sentiment_Analysis_MCQ_eval": "Sentiment Analysis"
}

# Models configuration
MODELS = {
    "ALLAM-7B": {
        "name": "ALLaM-AI/ALLaM-7B-Instruct-preview",
        "requires_auth": False,
        "load_in_8bit": True,
        "max_length": 2048
    },
    "SILMA-9B": {
        "name": "silma-ai/SILMA-9B-Instruct-v1.0",
        "requires_auth": False,
        "load_in_8bit": True,
        "max_length": 2048
    },
    "Fanar-1-9B-Instruct": {
        "name": "QCRI/Fanar-1-9B-Instruct",   
        "requires_auth": False,
            "load_in_8bit": True,
        "max_length": 2048,
        "tokenizer_kwargs": {"return_token_type_ids": False},
        "generate_kwargs": {"temperature": 0.0}
    },

    "Falcon-H1-7B-Instruct": {
        "name": "tiiuae/Falcon-H1-7B-Instruct",
        "requires_auth": False,
        "load_in_8bit": True,              
        "max_length": 2048,
        "tokenizer_kwargs": {"return_token_type_ids": False},
        "generate_kwargs": {"temperature": 0.0}
    },

    "Falcon-H1-1B-Base": {
        "name": "tiiuae/Falcon-H1-1B-Base",
        "requires_auth": False,
        "load_in_8bit": True,
        "max_length": 2048,
        "tokenizer_kwargs": {"return_token_type_ids": False},
        "generate_kwargs": {"temperature": 0.0}
    },
}


# Evaluation settings
BATCH_SIZE = 1
MAX_NEW_TOKENS = 10
MAX_SAMPLES_PER_DATASET = None  # e.g., 5 for quick test
RESULTS_DIR = "evaluation_results"

# Scratch & cache config
SCRATCH_BASE = "/scratch/rania.elbadry"
CACHE_DIR = os.path.join(SCRATCH_BASE, "transformers")

# Controls
CLEAR_CACHE_BEFORE_EACH_MODEL = True  # aggressively clear caches before every model

# ============================================================
# SETUP AND UTILITIES
# ============================================================

def set_env(visible_devices="0", scratch=SCRATCH_BASE):
    """
    Set environment variables for reliable runs and redirect caches to /scratch.
    Change visible_devices to "0,1,2,3,4" if you want multi-GPU sharding.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Prefer scratch caches to avoid home quota pressure
    if os.path.isdir("/scratch"):
        os.makedirs(f"{scratch}/huggingface", exist_ok=True)
        os.makedirs(f"{scratch}/transformers", exist_ok=True)
        os.makedirs(f"{scratch}/torch", exist_ok=True)
        os.environ["HF_HOME"] = f"{scratch}/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = f"{scratch}/transformers"
        os.environ["TORCH_HOME"] = f"{scratch}/torch"
    if HF_TOKEN:
        try:
            # Make sure both hub + datasets can see the token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
            os.environ["HF_TOKEN"] = HF_TOKEN

            login(token=HF_TOKEN, add_to_git_credential=False)
            print("✅ Logged in to HuggingFace (token set for hub & datasets)")
        except Exception as e:
            print(f"⚠️  HuggingFace login issue: {e}")
            print("   Will still try using HUGGINGFACE_HUB_TOKEN/HF_TOKEN if set.")
    else:
        print("ℹ️  HF_TOKEN not set; private datasets will fail to load.")

def check_disk_space(path="/"):
    """Check available disk space."""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)
    used_percent = (stat.used / stat.total) * 100

    print(f"💾 Disk Space ({path}): Free={free_gb:.2f} GB | Total={total_gb:.2f} GB | Used={used_percent:.1f}%")
    if free_gb < 50:
        print(f"⚠️  WARNING: Low disk space! Only {free_gb:.2f} GB free. Prefer /scratch.")
    return free_gb

def setup_multi_gpu():
    """Detect GPUs; return count. (We won’t force sharding here.)"""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"🖥️  Found {n_gpus} GPU(s):")
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {name} | Memory: {mem:.2f} GB")
        return n_gpus
    else:
        print("❌ No CUDA device found. Running on CPU.")
        return 0

def cleanup_memory():
    """Clean up CPU/GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def clear_disk_cache():
    """
    Clear HuggingFace/Torch disk caches (home AND scratch) to prevent quota issues.
    We clear contents of known cache dirs (safer than removing root).
    """
    cache_dirs = [
        os.getenv("HF_HOME"),
        os.getenv("TRANSFORMERS_CACHE"),
        os.getenv("TORCH_HOME"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/huggingface/transformers"),
        os.path.expanduser("~/.cache/torch/hub"),
        "/tmp/transformers_cache",
    ]

    print("🧹 Clearing disk caches...")
    for cache_dir in cache_dirs:
        if not cache_dir:
            continue
        if os.path.exists(cache_dir):
            try:
                for entry in os.listdir(cache_dir):
                    path = os.path.join(cache_dir, entry)
                    try:
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        else:
                            os.remove(path)
                    except Exception as e:
                        print(f"  ⚠️ Could not remove {path}: {e}")
                print(f"  ✅ Cleared contents of {cache_dir}")
            except Exception as e:
                print(f"  ⚠️ Could not clear {cache_dir}: {e}")

    # Clear common lock files
    lock_patterns = [
        os.path.expanduser("~/.cache/huggingface/**/*.lock"),
        "/tmp/huggingface_*.lock",
    ]
    for pat in lock_patterns:
        for lock_path in glob.glob(pat, recursive=True):
            try:
                if os.path.isdir(lock_path):
                    shutil.rmtree(lock_path, ignore_errors=True)
                else:
                    os.remove(lock_path)
            except Exception:
                pass

def print_memory_status(stage=""):
    """Print current memory usage for all GPUs (informational)."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i} Memory {stage}: Alloc={allocated:.2f}GB | Resv={reserved:.2f}GB")

def setup_environment():
    """Setup CUDA and authentication; redirect caches to scratch; create result dir."""
    print("="*60)
    print("ENVIRONMENT SETUP")
    print("="*60)

    # Set env (default to single GPU 0; change here if you want multi-GPU)
    set_env(visible_devices="0", scratch=SCRATCH_BASE)

    # Check disk space
    check_disk_space("/l/users/rania.elbadry")
    if os.path.isdir("/scratch"):
        os.makedirs(SCRATCH_BASE, exist_ok=True)
        print(f"✅ Using scratch directory: {SCRATCH_BASE}")
        check_disk_space("/scratch")

    # GPU info
    n_gpus = setup_multi_gpu()

    # HF login (optional if you use only public models)
    if HF_TOKEN:
        try:
            login(token=HF_TOKEN, add_to_git_credential=False)
            print("✅ Logged in to HuggingFace")
        except Exception as e:
            print(f"⚠️  HuggingFace login issue: {e}")
            print("   Continuing without login - will use public models only")
    else:
        print("ℹ️  HF_TOKEN not set; proceeding with public HF access.")

    # Results dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"✅ Results directory: {RESULTS_DIR}")

    return n_gpus

def move_inputs_to_model_device_if_single(model, inputs: dict):
    """
    If the model is on a single device (not sharded across >1 GPU),
    move tokenized inputs to that device to avoid CPU/GPU mismatch.
    If sharded, leave inputs on CPU so HF/Accelerate can route them.
    """
    try:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict):
            unique_devs = set(str(d) for d in device_map.values())
            if len(unique_devs) > 1:
                return inputs  
    except Exception:
        pass

    try:
        dev = model.get_input_embeddings().weight.device
    except Exception:
        try:
            dev = next(p.device for p in model.parameters() if p is not None)
        except Exception:
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}

def load_model_and_tokenizer(model_config, pretty_name, n_gpus=1):
    """Load model with correct handling for 8-bit quantization and device mapping."""
    print(f"\n{'='*60}")
    print(f"Loading: {pretty_name}")
    print(f"Path: {model_config['name']}")
    print(f"{'='*60}")

    cleanup_memory()
    print_memory_status("before loading")

    try:
        model_path = model_config['name']

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=HF_TOKEN if model_config.get('requires_auth') else None,
            trust_remote_code=True,
            padding_side='left',
            cache_dir=CACHE_DIR,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs = {
            "trust_remote_code": True,
            "cache_dir": CACHE_DIR,
            "token": HF_TOKEN if model_config.get('requires_auth') else None,
        }

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        multi = ("," in visible) and (n_gpus > 1)

        quantized = False
        if model_config.get("load_in_8bit", False):
            print("📊 Using 8-bit quantization via BitsAndBytesConfig")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            quantized = True

       
        if quantized:
            model_kwargs["device_map"] = "auto"
        else:
            torch_dtype = (
                torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                else torch.float16
            )
            model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs["device_map"] = "auto" if multi else None

        print("⏳ Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        if not quantized and model_kwargs.get("device_map") is None and torch.cuda.is_available():
            model.to("cuda")

        model.eval()
        print_memory_status("after loading")
        print(f"✅ Model loaded successfully ({'multi-GPU' if multi else 'single-GPU'}"
              f"{', 8-bit' if quantized else ''})")
        return model, tokenizer

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        cleanup_memory()
        return None, None

def create_prompt(question, model_name):
    """Create appropriate prompt for each model."""
    instruction = "أجب على السؤال التالي باختيار الحرف الصحيح فقط (a, b, c، أو d):"
    if "jais" in model_name.lower():
        return f"### Instruction:\n{instruction}\n\n### Question:\n{question}\n\n### Response:\n"
    elif "allam" in model_name.lower():
        return f"[INST] {instruction}\n\n{question} [/INST]\nالإجابة: "
    elif "silma" in model_name.lower():
        return f"<|user|>\n{instruction}\n\n{question}\n<|assistant|>\n"
    else:
        return f"{instruction}\n\n{question}\n\nالإجابة: "

def extract_answer(generated_text):
    """Extract MCQ answer (a/b/c/d) from generated text."""
    if not generated_text:
        return None
    text = generated_text.strip().lower()
    patterns = ['الإجابة:', 'الجواب:', 'الإجابة هي:', 'الإجابة الصحيحة:', 'الخيار الصحيح:', 'answer:', 'the answer is:']
    for p in patterns:
        if p in text:
            after = text.split(p)[-1].strip()
            for ch in after[:10]:
                if ch in ['a', 'b', 'c', 'd']:
                    return ch
    for ch in text[:20]:
        if ch in ['a', 'b', 'c', 'd']:
            return ch
    return None

def evaluate_single_example(model, tokenizer, query, correct_answer, model_name, model_config=None):
    try:
        prompt = create_prompt(query, model_name)

        tk_kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": 1024,
        }
        if model_config and "tokenizer_kwargs" in model_config:
            tk_kwargs.update(model_config["tokenizer_kwargs"])

        inputs = tokenizer(prompt, **tk_kwargs)

        inputs = move_inputs_to_model_device_if_single(model, inputs)

        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if model_config and "generate_kwargs" in model_config:
            gen_kwargs.update(model_config["generate_kwargs"])

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = generated_text[len(prompt):] if generated_text.startswith(prompt) else generated_text

        predicted = extract_answer(generated_part)
        is_correct = (predicted == correct_answer) if predicted else False

        return {
            'predicted': predicted,
            'correct': correct_answer,
            'is_correct': is_correct,
            'generated_text': generated_part[:200]
        }
    except Exception as e:
        print(f"   ⚠️  Error evaluating example: {e}")
        return {'predicted': None, 'correct': correct_answer, 'is_correct': False,
                'generated_text': f"Error: {str(e)}"}


def evaluate_dataset(model, tokenizer, dataset_name, model_name, model_config):
    """Evaluate model on a single dataset."""
    display_name = DATASET_DISPLAY_NAMES.get(
        dataset_name.split('/')[-1], dataset_name.split('/')[-1]
    )

    print(f"\n📊 Evaluating on: {display_name}")
    print(f"   Dataset: {dataset_name}")

    # ---- load dataset (private/gated ok) ----
    try:
        dataset = load_dataset(
            dataset_name,
            use_auth_token=HF_TOKEN,    # expects env var HF_TOKEN set
            cache_dir=CACHE_DIR
        )
        if 'test' in dataset:
            data, split = dataset['test'], 'test'
        elif 'validation' in dataset:
            data, split = dataset['validation'], 'validation'
        else:
            first = list(dataset.keys())[0]
            data, split = dataset[first], first

        print(f"   Split: {split} ({len(data)} examples)")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

    # Optional sample cap
    if MAX_SAMPLES_PER_DATASET:
        data = data.select(range(min(MAX_SAMPLES_PER_DATASET, len(data))))
        print(f"   Using {len(data)} samples for testing")

    # ---- per-model kwargs prepared once ----
    tk_kwargs = {"return_tensors": "pt", "truncation": True, "max_length": 1024}
    tk_kwargs.update(model_config.get("tokenizer_kwargs", {}))

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    gen_kwargs.update(model_config.get("generate_kwargs", {}))

    all_results, correct_count, total_count = [], 0, 0

    for idx in tqdm(range(len(data)), desc=f"   {display_name}"):
        # Be robust to slight schema differences
        row = data[idx]
        query = (
            row.get('query')
            or row.get('question')
            or row.get('prompt')
        )
        answer = (
            row.get('answer')
            or row.get('label')
            or row.get('gold')
        )

        if query is None or answer is None:
            all_results.append({
                'predicted': None, 'correct': answer, 'is_correct': False,
                'generated_text': 'Error: missing query/answer fields'
            })
            total_count += 1
            continue

        # Evaluate single example with per-model kwargs
        result = evaluate_single_example(
            model, tokenizer, query, answer, model_config['name'], model_config
        )
        all_results.append(result)
        if result['is_correct']:
            correct_count += 1
        total_count += 1

        if idx % 10 == 0 and idx > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
    print(f"   ✅ Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")

    return {
        'dataset': dataset_name.split('/')[-1],
        'display_name': display_name,
        'split': split,
        'total': total_count,
        'correct': correct_count,
        'accuracy': accuracy,
        'details': all_results
    }


def evaluate_model(model_key, model_config, n_gpus=1):
    """Evaluate a single model on all datasets."""
    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL: {model_key}")
    print(f"{'='*70}")

    # HARD CLEANUP before loading this model
    cleanup_memory()
    if CLEAR_CACHE_BEFORE_EACH_MODEL:
        clear_disk_cache()
    check_disk_space("/l/users/rania.elbadry")

    # Load
    model, tokenizer = load_model_and_tokenizer(model_config, model_key, n_gpus)
    if model is None:
        print(f"⚠️  Skipping {model_key} due to loading error")
        return None

    results = {
        'model': model_key,
        'model_path': model_config['name'],
        'timestamp': datetime.now().isoformat(),
        'dataset_results': []
    }
    for dataset_name in MCQ_DATASETS:
        try:
            dataset_result = evaluate_dataset(
                model,
                tokenizer,
                dataset_name,
                model_config['name'],   # model_name
                model_config            # <-- missing arg
            )
            if dataset_result:
                results['dataset_results'].append(dataset_result)
        except Exception as e:
            print(f"   ❌ Error on dataset {dataset_name}: {e}")
            cleanup_memory()
            continue


    # Cleanup model
    del model
    del tokenizer
    cleanup_memory()
    print_memory_status(f"after {model_key} cleanup")

    return results

# ============================================================
# RESULTS AND VISUALIZATION
# ============================================================

def create_comparison_tables(all_results):
    """Create detailed comparison tables."""
    rows = []
    for model_result in all_results:
        if not model_result:
            continue
        for dataset_result in model_result['dataset_results']:
            rows.append({
                'Model': model_result['model'],
                'Dataset': dataset_result['display_name'],
                'Accuracy': dataset_result['accuracy'],
                'Correct': dataset_result['correct'],
                'Total': dataset_result['total']
            })

    if not rows:
        print("⚠️  No data to create tables")
        return None, None

    df = pd.DataFrame(rows)
    pivot = df.pivot(index='Dataset', columns='Model', values='Accuracy')

    if not pivot.empty:
        pivot['Average'] = pivot.mean(axis=1, numeric_only=True)
        model_cols = [c for c in pivot.columns if c != 'Average']
        if model_cols:
            pivot['Best Model'] = pivot[model_cols].idxmax(axis=1)

        numeric_cols = pivot.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            avg_row = pivot[numeric_cols].mean()
            pivot.loc['AVERAGE'] = avg_row
            model_cols = [c for c in numeric_cols if c != 'Average']
            if model_cols:
                best_model = pivot.loc[:, model_cols].mean().idxmax()
                pivot.loc['AVERAGE', 'Best Model'] = best_model

    return df, pivot

def generate_report(all_results):
    """Generate comprehensive evaluation report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df, pivot = create_comparison_tables(all_results)
    if df is None or pivot is None:
        print("⚠️  Not enough data to generate report")
        return None, None

    json_path = os.path.join(RESULTS_DIR, f'evaluation_results_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(RESULTS_DIR, f'evaluation_comparison_{timestamp}.csv')
    pivot.to_csv(csv_path)

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    display_pivot = pivot.copy()
    for col in display_pivot.columns:
        if col not in ['Best Model']:
            display_pivot[col] = display_pivot[col].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) and isinstance(x, (int, float)) else x
            )

    print("\n📊 ACCURACY COMPARISON TABLE")
    print("-"*80)
    print(tabulate(display_pivot, headers='keys', tablefmt='grid'))

    print(f"\n📁 Results saved to:")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")

    return json_path, csv_path

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*80)
    print("ARABIC MODELS MCQ EVALUATION - CLEAN & STABLE")
    print("="*80)
    print(f"Models: {', '.join(MODELS.keys())}")
    print(f"Datasets: {len(MCQ_DATASETS)} tasks")
    print("="*80)

    # Environment & GPU info
    n_gpus = setup_environment()

    # Quick scratch/home check
    check_disk_space("/l/users/rania.elbadry")
    if os.path.isdir("/scratch"):
        check_disk_space("/scratch")

    # Start timer
    start_time = time.time()

    all_results = []

    # Warn if low space
    free_space = check_disk_space("/l/users/rania.elbadry")
    if free_space < 30:
        print("\n⚠️  Low disk space! Consider setting MAX_SAMPLES_PER_DATASET = 5 for a dry run.\n")

    for model_key, model_config in MODELS.items():
        try:
            result = evaluate_model(model_key, model_config, n_gpus)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error evaluating {model_key}: {e}")
            import traceback
            traceback.print_exc()
            cleanup_memory()
            continue

    if all_results:
        generate_report(all_results)
    else:
        print("\n❌ No results to report")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total execution time: {elapsed/60:.1f} minutes")

    cleanup_memory()
    if CLEAR_CACHE_BEFORE_EACH_MODEL:
        clear_disk_cache()
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    # Example: quick smoke test
    # MAX_SAMPLES_PER_DATASET = 5
    main()

"""
Utility functions for Arabic Accounting MCQ Evaluation
Based on the dialogue_steering utils pattern
"""

import os
import json
import yaml
import random
import math
import logging
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: Optional[str]) -> Optional[str]:
    """Create directory if it doesn't exist."""
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_logging(
    level: str = "INFO", format_str: str = "[%(levelname)s] %(asctime)s - %(message)s"
):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_existing_results(file_path: str) -> Dict[str, Any]:
    """Load existing results for resume functionality."""
    if not os.path.exists(file_path):
        return {}

    existing = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing[item["id"]] = item
    except Exception as e:
        logging.warning(f"Error loading existing results from {file_path}: {e}")

    return existing


def save_result(result: Dict[str, Any], file_path: str):
    """Save a single result to JSONL file."""
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def build_mcq_prompt(query: str, choices: List[str]) -> List[Dict[str, str]]:
    """
    Build chat messages for MCQ evaluation.
    The query already contains the full prompt with instructions, so pass it directly.
    """
    return [{"role": "user", "content": query}]


def build_generation_prompt(prompt: str) -> List[Dict[str, str]]:
    """
    Build chat messages for generation evaluation.
    The prompt already contains the full question/instruction, so pass it directly.
    """
    return [{"role": "user", "content": prompt}]


def parse_hf_dataset_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse HuggingFace dataset item to standardized format for MCQ.
    Expected input format:
    {
      "id": "unique_identifier",
      "query": "Full question with instructions in Arabic",
      "answer": "correct_letter",
      "text": "Question without instructions",
      "choices": ["a", "b", "c", "d"],
      "gold": 0  // Zero-based index of correct answer
    }
    """
    return {
        "id": item["id"],
        "query": item["query"],
        "text": item.get("text", ""),
        "choices": item["choices"],
        "correct_answer": item["answer"],
        "correct_index": item["gold"],
    }


def parse_generation_dataset_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse HuggingFace dataset item to standardized format for generation.
    Expected input format:
    {
      "id": "unique_identifier",
      "prompt": "Full question with instructions in Arabic",
      "question": "Question text (optional)",
      "answer": "Expected answer/response"
    }
    """
    return {
        "id": item["id"],
        "prompt": item["prompt"],
        "question": item.get("question", ""),
        "answer": item["answer"],
    }


def is_generation_dataset(item: Dict[str, Any]) -> bool:
    """
    Determine if a dataset item is for generation (vs MCQ) based on its structure.
    Generation datasets have 'prompt' key, MCQ datasets have 'query' and 'choices' keys.
    """
    return "prompt" in item and "choices" not in item

"""
Multi-dataset Zero-shot evaluation script for Arabic MCQ and Generation datasets
Evaluates multiple datasets with organized output structure
"""

import argparse
import os
import json
import logging
from typing import Dict, List, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

from utils import (
    load_yaml,
    ensure_dir,
    set_all_seeds,
    setup_logging,
    load_existing_results,
    save_result,
    build_mcq_prompt,
    build_generation_prompt,
    parse_hf_dataset_item,
    parse_generation_dataset_item,
    is_generation_dataset,
)
from api_client import create_api_client


def load_model(hf_id: str, max_len: int, four_bit: bool = True):
    """Load open source model for local inference."""
    logging.info(f"Loading model: {hf_id}")

    # Check if CUDA is available and only use quantization if it is
    use_cuda = torch.cuda.is_available()

    quant = None
    if four_bit and use_cuda:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif four_bit and not use_cuda:
        logging.warning(
            "4-bit quantization requires CUDA. Running without quantization."
        )

    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)

    # Use appropriate dtype based on hardware
    dtype = torch.bfloat16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        device_map="auto" if use_cuda else "cpu",
        dtype=dtype,
        quantization_config=quant,
    )

    tokenizer.model_max_length = max_len

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def generate_open_model(
    item: Dict[str, Any], tokenizer, model, generation_config: Dict[str, Any]
) -> str:
    """Generate response using open source model."""
    try:
        # Determine dataset type and build appropriate prompt
        if "query" in item and "choices" in item:
            # MCQ dataset
            messages = build_mcq_prompt(item["query"], item["choices"])
        elif "prompt" in item:
            # Generation dataset
            messages = build_generation_prompt(item["prompt"])
        else:
            raise ValueError(
                "Unknown dataset format: missing 'query'+'choices' or 'prompt' fields"
            )

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=generation_config["max_new_tokens"],
                temperature=generation_config["temperature"],
                top_p=generation_config["top_p"],
                do_sample=generation_config["do_sample"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        return response

    except Exception as e:
        logging.error(f"Error generating response for item {item['id']}: {e}")
        return ""


def generate_closed_model(
    item: Dict[str, Any],
    model_config: Dict[str, Any],
    api_client,
    generation_config: Dict[str, Any],
) -> str:
    """Generate response using closed model via API."""
    try:
        # Determine dataset type and build appropriate prompt
        if "query" in item and "choices" in item:
            print("mcq dataset detected")
            # MCQ dataset
            messages = build_mcq_prompt(item["query"], item["choices"])
        elif "prompt" in item:
            print("generation dataset detected")
            # Generation dataset
            messages = build_generation_prompt(item["prompt"])
        else:
            raise ValueError(
                "Unknown dataset format: missing 'query'+'choices' or 'prompt' fields"
            )

        response = api_client.generate(
            model_name=model_config["model_name"],
            messages=messages,
            temperature=generation_config["temperature"],
            max_tokens=generation_config["max_new_tokens"],
            provider=model_config.get(
                "provider", "deepinfra"
            ),  # Default to deepinfra for backward compatibility
        )

        return response

    except Exception as e:
        logging.error(
            f"Error generating response for item {item['id']} with {model_config['model_name']}: {e}"
        )
        return ""


def evaluate_split_open_model(
    dataset_split,
    model_alias: str,
    model_config: Dict[str, Any],
    generation_config: Dict[str, Any],
    output_file: str,
    resume_enabled: bool = True,
):
    """Evaluate dataset split using open source model."""

    # Load existing results for resume
    existing_results = {}
    if resume_enabled:
        existing_results = load_existing_results(output_file)
        logging.info(f"Loaded {len(existing_results)} existing results for resume")

    # Load model
    tokenizer, model = load_model(
        model_config["hf_id"],
        model_config["max_len"],
        model_config.get("four_bit", True),
    )

    # Process each item
    for item in tqdm(dataset_split, desc=f"Evaluating {model_alias}"):
        # Determine dataset type and parse accordingly
        if is_generation_dataset(item):
            parsed_item = parse_generation_dataset_item(item)
        else:
            parsed_item = parse_hf_dataset_item(item)

        # Skip if already processed
        if resume_enabled and parsed_item["id"] in existing_results:
            continue

        # Generate response
        response = generate_open_model(parsed_item, tokenizer, model, generation_config)

        # Create result record - only add model_response to original item
        result = {
            **item,  # Keep all original dataset fields
            "model_response": response,
        }

        # Save result
        save_result(result, output_file)

    logging.info(
        f"Completed evaluation for {model_alias}, results saved to {output_file}"
    )


def evaluate_split_closed_model(
    dataset_split,
    model_alias: str,
    model_config: Dict[str, Any],
    generation_config: Dict[str, Any],
    api_client,
    output_file: str,
    resume_enabled: bool = True,
):
    """Evaluate dataset split using closed model via API."""

    # Load existing results for resume
    existing_results = {}
    if resume_enabled:
        existing_results = load_existing_results(output_file)
        logging.info(f"Loaded {len(existing_results)} existing results for resume")

    # Process each item
    for item in tqdm(dataset_split, desc=f"Evaluating {model_alias}"):
        # Determine dataset type and parse accordingly
        if is_generation_dataset(item):
            print("generation dataset detected")
            parsed_item = parse_generation_dataset_item(item)
        else:
            print("mcq dataset detected")
            parsed_item = parse_hf_dataset_item(item)

        # Skip if already processed
        if resume_enabled and parsed_item["id"] in existing_results:
            continue

        # Generate response
        response = generate_closed_model(
            parsed_item, model_config, api_client, generation_config
        )

        # Create result record - only add model_response to original item
        result = {
            **item,  # Keep all original dataset fields
            "model_response": response,
        }

        # Save result
        save_result(result, output_file)

    logging.info(
        f"Completed evaluation for {model_alias}, results saved to {output_file}"
    )


def evaluate_dataset(
    dataset_config: Dict[str, Any],
    models_config: Dict[str, Any],
    generation_config: Dict[str, Any],
    base_output_dir: str,
    api_client,
    model_type: str,
    model_filter: List[str] = None,
    split_filter: List[str] = None,
    resume_enabled: bool = True,
):
    """Evaluate a single dataset with all specified models."""

    dataset_name = dataset_config["name"]
    hf_name = dataset_config["hf_name"]

    logging.info(f"\n{'='*60}")
    logging.info(f"Starting evaluation for dataset: {dataset_name}")
    logging.info(f"HuggingFace dataset: {hf_name}")
    logging.info(f"{'='*60}")

    # Load dataset
    try:
        dataset = load_dataset(hf_name)
        logging.info(f"Dataset loaded successfully with splits: {list(dataset.keys())}")
    except Exception as e:
        logging.error(f"Failed to load dataset {hf_name}: {e}")
        return

    # Filter splits
    splits_to_eval = dataset_config["splits"]
    if split_filter:
        splits_to_eval = [s for s in splits_to_eval if s in split_filter]

    # Create dataset-specific output directory
    dataset_output_dir = os.path.join(base_output_dir, dataset_name)
    ensure_dir(dataset_output_dir)

    # Evaluate open models
    if model_type in ["open", "all"]:
        open_models = models_config.get("eval_models", {}).get("open", [])
        if model_filter:
            open_models = [m for m in open_models if m in model_filter]

        if open_models:
            logging.info(
                f"Starting evaluation of {len(open_models)} open models: {open_models}"
            )

            for model_alias in open_models:
                if model_alias not in models_config["open_models"]:
                    logging.warning(
                        f"Model {model_alias} not found in open_models config"
                    )
                    continue

                model_config = models_config["open_models"][model_alias]

                for split in splits_to_eval:
                    if split not in dataset:
                        logging.warning(
                            f"Split {split} not found in dataset {dataset_name}"
                        )
                        continue

                    output_file = os.path.join(
                        dataset_output_dir,
                        f"{split}_{model_alias}_results.jsonl",
                    )

                    evaluate_split_open_model(
                        dataset[split],
                        model_alias,
                        model_config,
                        generation_config,
                        output_file,
                        resume_enabled,
                    )

    # Evaluate closed models
    if model_type in ["closed", "all"] and api_client:
        closed_models = models_config.get("eval_models", {}).get("closed", [])
        if model_filter:
            closed_models = [m for m in closed_models if m in model_filter]

        if closed_models:
            logging.info(
                f"Starting evaluation of {len(closed_models)} closed models: {closed_models}"
            )

            for model_alias in closed_models:
                if model_alias not in models_config["closed_models"]:
                    logging.warning(
                        f"Model {model_alias} not found in closed_models config"
                    )
                    continue

                model_config = models_config["closed_models"][model_alias]

                for split in splits_to_eval:
                    if split not in dataset:
                        logging.warning(
                            f"Split {split} not found in dataset {dataset_name}"
                        )
                        continue

                    output_file = os.path.join(
                        dataset_output_dir,
                        f"{split}_{model_alias}_results.jsonl",
                    )

                    evaluate_split_closed_model(
                        dataset[split],
                        model_alias,
                        model_config,
                        generation_config,
                        api_client,
                        output_file,
                        resume_enabled,
                    )

    logging.info(f"Completed evaluation for dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-dataset Zero-shot evaluation for Arabic MCQ and Generation datasets"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--models", required=True, help="Path to models.yaml")
    parser.add_argument(
        "--model-type",
        choices=["open", "closed", "all"],
        default="all",
        help="Type of models to evaluate",
    )
    parser.add_argument(
        "--model-filter", help="Comma-separated list of specific models to run"
    )
    parser.add_argument(
        "--split-filter", help="Comma-separated list of specific splits to run"
    )
    parser.add_argument(
        "--dataset-filter", help="Comma-separated list of specific datasets to run"
    )

    args = parser.parse_args()

    # Load configurations
    config = load_yaml(args.config)
    models_config = load_yaml(args.models)

    # Setup
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    set_all_seeds(config.get("seed", 42))

    # Create base output directory
    base_output_dir = config["base_config"]["outputs_dir"]
    ensure_dir(base_output_dir)

    # Filter datasets
    datasets_to_eval = config["datasets"]
    if args.dataset_filter:
        requested_datasets = [d.strip() for d in args.dataset_filter.split(",")]
        datasets_to_eval = [
            d for d in datasets_to_eval if d["name"] in requested_datasets
        ]

    # Parse filters
    model_filter = None
    if args.model_filter:
        model_filter = [m.strip() for m in args.model_filter.split(",")]

    split_filter = None
    if args.split_filter:
        split_filter = [s.strip() for s in args.split_filter.split(",")]

    # Create API client for closed models
    api_client = None
    if args.model_type in ["closed", "all"]:
        try:
            api_client = create_api_client(config)
            logging.info("API client created successfully for closed models")
        except Exception as e:
            logging.warning(f"Failed to create API client: {e}")
            if args.model_type == "closed":
                logging.error("API client required for closed models only. Exiting.")
                return
            else:
                logging.warning(
                    "API client failed, will skip closed models and only run open models"
                )

    # Evaluate each dataset
    total_datasets = len(datasets_to_eval)
    logging.info(f"Starting multi-dataset evaluation for {total_datasets} datasets")

    for i, dataset_config in enumerate(datasets_to_eval, 1):
        logging.info(
            f"\nProcessing dataset {i}/{total_datasets}: {dataset_config['name']}"
        )

        evaluate_dataset(
            dataset_config=dataset_config,
            models_config=models_config,
            generation_config=config["generation"],
            base_output_dir=base_output_dir,
            api_client=api_client,
            model_type=args.model_type,
            model_filter=model_filter,
            split_filter=split_filter,
            resume_enabled=config.get("resume", {}).get("enabled", True),
        )

    # Final summary
    logging.info(f"\n{'='*60}")
    logging.info("MULTI-DATASET EVALUATION COMPLETED")
    logging.info(f"{'='*60}")

    # open_count = len(models_config.get("eval_models", {}).get("open", []))
    # closed_count = len(models_config.get("eval_models", {}).get("closed", []))

    # if args.model_type == "all":
    #     if api_client:
    #         logging.info(
    #             f"Evaluated {total_datasets} datasets with {open_count} open + {closed_count} closed models"
    #         )
    #     else:
    #         logging.info(
    #             f"Evaluated {total_datasets} datasets with {open_count} open models only"
    #         )
    # elif args.model_type == "open":
    #     logging.info(
    #         f"Evaluated {total_datasets} datasets with {open_count} open models"
    #     )
    # elif args.model_type == "closed":
    #     logging.info(
    #         f"Evaluated {total_datasets} datasets with {closed_count} closed models"
    #     )

    logging.info(f"Results saved to: {base_output_dir}")
    logging.info("Directory structure:")
    for dataset_config in datasets_to_eval:
        logging.info(f"  └── {dataset_config['name']}/")


if __name__ == "__main__":
    main()

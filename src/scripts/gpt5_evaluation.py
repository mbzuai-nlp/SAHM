import os
import json
import time
from typing import Optional, List, Dict
from datasets import load_dataset
from openai import OpenAI

# Configuration
MODEL = "gpt-5"  # or "openai/gpt-4o-mini" for cost optimization
BATCH_SIZE = 5  # Process 5 entries at a time

# HF Dataset configuration
DATASET_NAME = "SahmBenchmark/Islamic_Finance_QnA_eval"

# OpenRouter client
client = OpenAI(
    api_key="your_api_key_here",
    base_url="https://api.openai.com/v1",
)


def query_gpt5_batch(
    model: str,
    batch_requests: List[str],
    retries: int = 5,
    wait_time: int = 8,
) -> List[str]:
    """
    Process multiple text requests in batch.
    batch_requests: List of prompt strings
    """
    results = []

    for prompt in batch_requests:
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    # max_output_tokens=4096,
                    # temperature=0.0,
                    top_p=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    reasoning_effort="minimal",
                )
                result = (response.choices[0].message.content or "").strip()
                results.append(result)
                break
            except Exception as e:
                print(f"[Retry {attempt+1}/{retries}] Error: {type(e).__name__} - {e}")
                if attempt == retries - 1:
                    results.append("CONNECTION_FAILED")
                else:
                    time.sleep(wait_time)

    return results


def query_gpt5(
    model: str,
    prompt: str,
    retries: int = 5,
    wait_time: int = 8,
) -> str:
    """
    Query GPT-5 model with a single prompt
    """
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_output_tokens=2048,
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                reasoning_effort="minimal",  # Cost optimization
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[Retry {attempt+1}/{retries}] Error: {type(e).__name__} - {e}")
            time.sleep(wait_time)
    return "CONNECTION_FAILED"


def process_dataset(
    dataset_name: str,
    split_name: str,
    out_path: str,
    max_samples: Optional[int] = None,
):
    """
    Process the Arabic accounting MCQ dataset for a specific split
    """
    # Resume-safe load for JSONL format
    processed_ids = set()
    existing_data = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf8") as f:
            try:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        existing_data.append(item)
                        processed_ids.add(
                            str(
                                item.get(
                                    "id", item.get("index", len(existing_data) - 1)
                                )
                            )
                        )
                print(
                    f"Found {len(processed_ids)} already processed entries. Resuming..."
                )
            except Exception as e:
                print(f"Error reading existing file: {e}")
                existing_data = []

    # Load dataset from HuggingFace
    try:
        print(f"Loading dataset: {dataset_name}, split: {split_name}")
        dataset = load_dataset(dataset_name)

        if split_name not in dataset:
            raise ValueError(
                f"Split '{split_name}' not found in dataset. Available splits: {list(dataset.keys())}"
            )

        data_split = dataset[split_name]

        if max_samples and max_samples > 0:
            samples = list(data_split.take(max_samples))
        else:
            samples = list(data_split)

        if not samples:
            raise ValueError(f"No samples found in dataset {dataset_name}.")

        print(f"Loaded {len(samples)} samples from dataset")

    except Exception as e:
        raise ValueError(f"Error loading dataset {dataset_name}: {e}")

    # Process entries in batches
    new_results = []

    for batch_start in range(0, len(samples), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(samples))
        batch_entries = samples[batch_start:batch_end]

        print(
            f"Processing batch {batch_start//BATCH_SIZE + 1}: entries {batch_start+1}-{batch_end}"
        )

        # Prepare batch data
        batch_prompts = []
        batch_data = []

        for i, entry in enumerate(batch_entries):
            # Generate a unique ID for the entry
            entry_id = str(entry.get("id", entry.get("index", batch_start + i)))

            # Skip if already processed
            if entry_id in processed_ids:
                continue

            # Get the query/prompt from the entry
            query = entry.get("prompt", entry.get("query", entry.get("ques", "")))

            if not query:
                print(f"Warning: No query found for entry {entry_id}")
                continue

            batch_prompts.append(query)
            batch_data.append((entry, entry_id))

        if not batch_prompts:
            continue

        # Process batch with GPT-5
        print(f"Sending {len(batch_prompts)} prompts to GPT-5...")
        batch_results = query_gpt5_batch(MODEL, batch_prompts)

        # Process results and update data
        for batch_idx, (entry, entry_id) in enumerate(batch_data):
            if batch_idx >= len(batch_results):
                break

            result = batch_results[batch_idx]

            print(f"Entry ID: {entry_id}")
            print(f"Query: {batch_prompts[batch_idx]}")
            print(f"Model Response: {result}")
            print("---")

            # Create new entry with all original keys plus model_response
            new_entry = dict(entry)
            new_entry["model_response"] = result

            # Add entry ID if not present
            if "id" not in new_entry and "index" not in new_entry:
                new_entry["id"] = entry_id

            new_results.append(new_entry)
            processed_ids.add(entry_id)

        # Save progress after each batch in JSONL format
        with open(out_path, "w", encoding="utf-8") as f:
            # Write existing data first
            for item in existing_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            # Write new results
            for item in new_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Processing completed. Total entries processed: {len(processed_ids)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GPT-5 Arabic Accounting MCQ Evaluation Pipeline"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max samples to process per split"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["test", "validation", "both"],
        help="Which split to process (default: both)",
    )

    args = parser.parse_args()

    print("\n=== GPT-5 Arabic Accounting MCQ Evaluation Pipeline ===")
    print(f"  Model           : {MODEL}")
    print(f"  Dataset         : {DATASET_NAME}")
    print(f"  Max Samples     : {args.max_samples or 'All'} (per split)")
    print(f"  Batch Size      : {BATCH_SIZE}")
    print(f"  Split(s)        : {args.split}")
    print("========================================================\n")

    # Determine which splits to process
    splits_to_process = []
    if args.split == "both":
        splits_to_process = ["test", "validation"]
    else:
        splits_to_process = [args.split]

    success_count = 0
    for split_name in splits_to_process:
        output_file = f"{split_name}_gpt5.jsonl"
        print(f"\n--- Processing {split_name} split ---")

        try:
            process_dataset(
                DATASET_NAME,
                split_name,
                output_file,
                args.max_samples,
            )
            print(f"✅ {split_name} split completed! Results saved to: {output_file}")
            success_count += 1
        except Exception as e:
            print(f"❌ Error processing {split_name} split: {e}")
            continue

    if success_count == len(splits_to_process):
        print(f"\n🎉 All splits processed successfully!")
        return 0
    else:
        print(
            f"\n⚠️  {success_count}/{len(splits_to_process)} splits processed successfully."
        )
        return 1


if __name__ == "__main__":
    exit(main())

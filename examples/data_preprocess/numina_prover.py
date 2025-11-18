# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the NuminaMath-LEAN dataset to parquet format
Extracts only LEAN4 theorem statements for training/testing.
Reward evaluation is done via Kimina Lean REPL server (1 for correct, 0 for incorrect).
"""

import argparse
import json
import os

import datasets
from datasets import Dataset
from transformers import AutoTokenizer

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="The local path to the raw dataset, if it exists.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/numina-prover",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument(
        "--test_split_ratio",
        type=float,
        default=0.2,
        help="Ratio of data to use for test split (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model for tokenizer (required for token filtering). Default: ~/Qwen/Qwen3-1.7B",
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens in prompt (theorem + instruction). Examples exceeding this will be filtered. Default: 2048",
    )
    parser.add_argument(
        "--filter_long_prompts",
        action="store_true",
        help="Enable filtering of prompts exceeding max_prompt_tokens. Requires --model_path.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # Load tokenizer if filtering is enabled
    tokenizer = None
    if args.filter_long_prompts:
        if args.model_path is None:
            args.model_path = os.path.expanduser("~/Qwen/Qwen3-1.7B")
        model_path = os.path.expanduser(args.model_path)
        print(f"Loading tokenizer from {model_path}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"Tokenizer loaded. Max prompt tokens: {args.max_prompt_tokens}", flush=True)

    data_source = "AI-MO/NuminaMath-LEAN"
    print(f"Loading the {data_source} dataset...", flush=True)

    if local_dataset_path is not None:
        # Load from local directory (after snapshot_download)
        dataset = datasets.load_dataset(
            local_dataset_path,
        )
    else:
        # Load directly from HuggingFace
        dataset = datasets.load_dataset(data_source)

    # Extract only the 'formal_statement' field (LEAN4 theorem statements)
    print("Extracting formal_statement fields...", flush=True)
    
    # Get the dataset split - NuminaMath-LEAN might have train/test or just one split
    if "train" in dataset:
        full_dataset = dataset["train"]
    elif "train" not in dataset and len(dataset) == 1:
        # If there's only one split, use it
        split_name = list(dataset.keys())[0]
        full_dataset = dataset[split_name]
    else:
        raise ValueError(f"Unexpected dataset structure: {list(dataset.keys())}")

    print(f"Total examples: {len(full_dataset)}", flush=True)

    # Split into train and test
    if args.test_split_ratio > 0:
        split_dataset = full_dataset.train_test_split(
            test_size=args.test_split_ratio, seed=args.seed
        )
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
    else:
        # Use all data for training, create empty test set
        train_dataset = full_dataset
        test_dataset = Dataset.from_dict({"formal_statement": []})

    print(f"Train examples: {len(train_dataset)}", flush=True)
    print(f"Test examples: {len(test_dataset)}", flush=True)

    instruction_following = "Prove this theorem in Lean 4."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # Extract the formal_statement (LEAN4 theorem)
            theorem_statement = example.get("formal_statement", "")
            
            if not theorem_statement:
                # Skip if no formal_statement found
                return None

            # Combine theorem statement with instruction
            prompt_content = theorem_statement
            if instruction_following:
                prompt_content = theorem_statement + "\n\n" + instruction_following

            # Filter by token count if tokenizer is available
            if tokenizer is not None:
                tokens = tokenizer.encode(prompt_content, add_special_tokens=False)
                token_count = len(tokens)
                if token_count > args.max_prompt_tokens:
                    # Return None to filter out this example
                    return None

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt_content}],
                "ability": "lean4",
                # No ground_truth - correctness evaluated by Kimina (1 for correct, 0 for incorrect)
                "reward_model": {"style": "tool", "ground_truth": None},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "theorem_statement": theorem_statement,
                },
            }
            return data

        return process_fn

    print("Processing train dataset...", flush=True)
    train_before_count = len(train_dataset)
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    # Filter out None values (if any examples had missing formal_statement or exceeded token limit)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    train_after_count = len(train_dataset)
    train_filtered = train_before_count - train_after_count
    if train_filtered > 0:
        print(f"Filtered out {train_filtered} train examples ({train_filtered/train_before_count*100:.2f}%) exceeding {args.max_prompt_tokens} tokens", flush=True)

    print("Processing test dataset...", flush=True)
    if len(test_dataset) > 0:
        test_before_count = len(test_dataset)
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
        test_dataset = test_dataset.filter(lambda x: x is not None)
        test_after_count = len(test_dataset)
        test_filtered = test_before_count - test_after_count
        if test_filtered > 0:
            print(f"Filtered out {test_filtered} test examples ({test_filtered/test_before_count*100:.2f}%) exceeding {args.max_prompt_tokens} tokens", flush=True)
        if len(test_dataset) == 0:
            print("Warning: All test examples were filtered out", flush=True)
    else:
        print("Warning: Test dataset is empty (test_split_ratio may be 0 or all data used for training)", flush=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Saving to {local_dir}...", flush=True)
    import pyarrow.parquet as pq
    
    # Save train dataset
    train_path = os.path.join(local_dir, "train.parquet")
    print(f"Saving train dataset ({len(train_dataset)} examples)...", flush=True)
    train_dataset.to_parquet(train_path)
    print(f"Saved train.parquet", flush=True)
    
    # Verify train parquet file
    try:
        schema = pq.read_schema(train_path)
        table = pq.read_table(train_path)
        print(f"Verified train.parquet is valid ({len(table)} rows)", flush=True)
    except Exception as e:
        print(f"ERROR: train.parquet appears corrupted: {e}", flush=True)
        if os.path.exists(train_path):
            os.remove(train_path)
        raise
    
    # Only save test dataset if it has examples
    if len(test_dataset) > 0:
        test_path = os.path.join(local_dir, "test.parquet")
        print(f"Saving test dataset ({len(test_dataset)} examples)...", flush=True)
        test_dataset.to_parquet(test_path)
        print(f"Saved test.parquet", flush=True)
        
        # Verify test parquet file
        try:
            schema = pq.read_schema(test_path)
            table = pq.read_table(test_path)
            print(f"Verified test.parquet is valid ({len(table)} rows)", flush=True)
        except Exception as e:
            print(f"ERROR: test.parquet appears corrupted: {e}", flush=True)
            if os.path.exists(test_path):
                os.remove(test_path)
            raise
    else:
        print("Warning: Test dataset is empty, skipping test.parquet creation", flush=True)

    # Save one example as JSON for reference
    if len(train_dataset) > 0:
        example = train_dataset[0]
        with open(os.path.join(local_dir, "train_example.json"), "w") as f:
            json.dump(example, f, indent=2)
    
    if len(test_dataset) > 0:
        example = test_dataset[0]
        with open(os.path.join(local_dir, "test_example.json"), "w") as f:
            json.dump(example, f, indent=2)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print("Done!", flush=True)
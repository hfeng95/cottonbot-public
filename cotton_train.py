#=========================================================
#   train_cotton.py
#
# Offline fine-tuning script for Cottonbot
# Uses Hugging Face Transformers
#=========================================================

import os
import argparse
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset

def load_dataset_from_source(path, tokenizer, block_size=256, text_column=None, split='train'):
    """Load source as a dataset."""

    # if file exist locally
    if os.path.isfile(path):
        if path.endswith(".txt"):
            return TextDataset(
                tokenizer=tokenizer,
                file_path=path,
                block_size=block_size
            )
        elif path.endswith('.csv'):
            raise Exception('Local csv not yet implemented.')
    
    # otherwise, load it via huggingface
    raw_data = load_dataset(path)

    # Handle DatasetDict vs Dataset
    if isinstance(raw_data, dict) or hasattr(raw_data, "keys"):
        if split in raw_data:
            dataset = raw_data[split]
        else:
            # Fallback: just take the first split if split not found
            first_split = list(raw_data.keys())[0]
            dataset = raw_data[first_split]
    else:
        dataset = raw_data

    # Determine which column to use
    if not text_column:
        sample_cols = dataset.column_names
        text_column = "text" if "text" in sample_cols else sample_cols[0]

    # if column contains lists, assume conversations
    if isinstance(dataset[text_column][0],list):
        def tokenize_function(examples):
            return tokenizer.apply_chat_template(
                examples[text_column],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=block_size
            )

    # Tokenize
    else:
        def tokenize_function(examples):
            return tokenizer(examples[text_column], truncation=True, max_length=block_size)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset

def fine_tune(train_path, output_dir, base_model="gpt2", steps=100, epochs=1, batch_size=1):
    """Fine-tune a model on a text dataset."""
    # if not os.path.exists(train_path):
    #     raise FileNotFoundError(f"Training file not found: {train_path}")

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)

    print(f"Loaded base model: {base_model}")
    print(f"Training on: {train_path}")
    print(f"Saving to: {output_dir}")

    dataset = load_dataset_from_source(train_path, tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=max(1, steps // 10),
        save_total_limit=1,
        logging_dir="./logs",
        max_steps=steps,
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    print("Beginning fine-tuning...")
    trainer.train()
    print("Training complete.")

    print("Saving model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("All done! Model saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for Cottonbot.")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to training text file (.txt)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--model", type=str, default="gpt2",
        help="Base model name (default: gpt2)"
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of training steps (default: 100)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Batch size per device (default: 1)"
    )

    args = parser.parse_args()

    fine_tune(
        train_path=args.data,
        output_dir=args.output,
        base_model=args.model,
        steps=args.steps,
        epochs=args.epochs,
        batch_size=args.batch
    )

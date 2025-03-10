import logging
import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import os
from sagemaker.huggingface import HuggingFace
import sagemaker
import boto3
import subprocess
import importlib


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def load_qa_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def process_qa_to_prompt(data):
    examples = []
    for entry in data:
        question = entry['instruction']
        answer = entry['response']
        prompt = f"Question: {question}\nAnswer: {answer}\n"
        examples.append({"text": prompt})
    return Dataset.from_list(examples)

def main():
    model_name = os.environ.get('SM_HP_MODEL_NAME_OR_PATH', 'meta-llama/Llama-2-7b-hf')
    train_file = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/synthetic_qa.jsonl')
    output_dir = os.environ.get('SM_OUTPUT_DIR', '/opt/ml/model')

    data = load_qa_data(train_file)
    dataset = process_qa_to_prompt(data)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # LoRA Configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=512, truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=3,
        fp16=True,
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="no",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()

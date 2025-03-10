import argparse
import logging
import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import os

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="Path to the pretrained model")
    parser.add_argument("--train_file", type=str, default="synthetic_qa.jsonl", 
                        help="Path to the training data file")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model", 
                        help="Directory to save the model")
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = args.model_name_or_path
    train_file = args.train_file
    output_dir = args.output_dir

    data = load_qa_data(train_file)
    dataset = process_qa_to_prompt(data)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # LoRA Config
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

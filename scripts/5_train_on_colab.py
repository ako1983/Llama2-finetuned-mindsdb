import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# paths 
DATA_PATH = "/content/synthetic_qa.jsonl"  #  Or google Drive 
OUTPUT_DIR = "/content/llama2-finetune/"   # Or google Drive 

# Load dataset
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

# Load and process dataset
data = load_qa_data(DATA_PATH)
dataset = process_qa_to_prompt(data)

# Load Model & Tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=512, truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=3,
    fp16=True,  # Enable mixed precision
    save_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    evaluation_strategy="no",
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets
)

# Train the model
trainer.train()

# Save Model & Tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to: {OUTPUT_DIR}")

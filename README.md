# Automated Pipeline

This repository includes an automated pipeline that:

1. Clones the MindsDB documentation repository.
2. Navigates to the `scripts` directory.
3. Executes the specified scripts in sequence.

## Running the Pipeline Manually

To run the pipeline locally:

```bash
bash ./scripts/run_pipeline.sh


# llama2-finetuned-mindsdb
This model is a fine-tuned version of `meta-llama/Llama-2-7b-hf` on the MindsDB documentation dataset.

## Training Details
Base Model: LLaMA-2 7B <br> 
Dataset: MindsDB documentation <br> 
Fine-tuning Method: LoRA <br> 
Training Epochs: 3 <br> 
Hardware: Google Colab Pro (A100 GPU) <br> 

## Usage
```python

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "ako-oak/llama2-finetuned-mindsdb"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(chat("What is the purpose of handlers in MindsDB?"))

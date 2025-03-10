from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load your fine-tuned model from HF
MODEL_NAME = "ako-oak/llama2-finetuned-mindsdb"  # Use your correct HF model name
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Run inference
def chat(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=200, temperature=0.7, top_p=0.9)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test it
print(chat("What is the purpose of handlers in MindsDB?"))

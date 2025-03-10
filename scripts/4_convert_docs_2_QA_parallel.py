import os
import json
import time
from tqdm import tqdm
from openai import OpenAI

# === Load OpenAI API Key from File ===
def load_openai_key(filepath="openai_api_key_QA.txt"):
    with open(filepath, "r", encoding="utf-8") as file:
        key = file.read().strip()
        os.environ["OPENAI_API_KEY"] = key
        return key

# Load API Key into environment and create OpenAI client
openai_key = load_openai_key()
client = OpenAI(api_key=openai_key)

# === File Paths ===
INPUT_FILE = "chunked_docs.json"
OUTPUT_FILE = "synthetic_qa.jsonl"

# === Helper Function to Call GPT-4 Turbo and Generate Q&A Pairs ===
def generate_qa_pairs(chunk_content, max_pairs=3, retries=3):
    system_prompt = (
        "You are a helpful AI assistant that generates synthetic question-answer pairs "
        "from provided documentation content. Please generate concise, relevant questions "
        "that a user might ask about the content, and provide accurate, to-the-point answers."
    )

    user_prompt = f"""
Here is a chunk of technical documentation:

--- START DOCUMENTATION CHUNK ---
{chunk_content}
--- END DOCUMENTATION CHUNK ---

Please generate up to {max_pairs} relevant question-answer pairs. Format the response as a JSON array like this:

[
    {{"question": "First question", "answer": "First answer"}},
    {{"question": "Second question", "answer": "Second answer"}}
]

If you cannot generate {max_pairs}, return fewer. Never generate more than {max_pairs}.
"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo", # or any other LLM#
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )

            content_text = response.choices[0].message.content
            qa_pairs = json.loads(content_text)

            return qa_pairs

        except Exception as e:
            print(f"OpenAI API error on attempt {attempt + 1}: {e}")
            time.sleep(5)

    print(f"Failed to generate Q&A pairs after {retries} attempts.")
    return []

# === Main Pipeline to Process All Chunks and Write synthetic_qa.jsonl ===
def generate_synthetic_qa():
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        data = json.load(infile)

        for chunk in tqdm(data, desc="Processing chunks"):
            filename = chunk["filename"]
            chunk_id = chunk["chunk_id"]
            chunk_content = chunk["chunk_content"]
            path = chunk.get("path", "")

            qa_pairs = generate_qa_pairs(chunk_content)

            for pair in qa_pairs:
                output_record = {
                    "instruction": pair["question"],
                    "response": pair["answer"],
                    "metadata": {
                        "filename": filename,
                        "chunk_id": chunk_id,
                        "path": path
                    }
                }
                outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    print(f"Synthetic Q&A pairs saved to {OUTPUT_FILE}")


generate_synthetic_qa()

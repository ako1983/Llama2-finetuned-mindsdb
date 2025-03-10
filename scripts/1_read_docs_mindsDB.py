import os
import json


# Define dir
DOCS_DIR = os.path.expanduser("docs")

# List all docs
def list_doc_files(directory, extensions=(".md", ".mdx")):
    doc_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                doc_files.append(os.path.join(root, file))
    return doc_files

# Function to read files
def read_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    # Process documentation files
def process_docs():
    if not os.path.exists(DOCS_DIR):
        print(f"Docs directory not found: {DOCS_DIR}")
        return
    
    print("📂 Locating documentation files...")
    doc_files = list_doc_files(DOCS_DIR)
    print(f"✅ Found {len(doc_files)} documentation files.")

    docs_data = []
    
    for filepath in doc_files:
        content = read_file(filepath)
        if content:
            docs_data.append({
                "filename": os.path.basename(filepath),
                "path": filepath,
                "content": content
            })
    
    # Save as JSON 
    output_file = os.path.join(DOCS_DIR, "docs_data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(docs_data, f, indent=4)
    
    print(f"✅ Processed and saved to {output_file}")

    ######
    if __name__ == "__main__":
        process_docs()
    ####

import json
import re
import nltk

# Downloading NLTK 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load the clean docs
with open("cleaned_docs.json", "r") as f:
    docs = json.load(f)

chunked_docs = []
# Process each doc
for doc in docs:
    # Chunk it
    chunks = chunk_text(doc["cleaned_content"], max_tokens=200)
    # Save each chunk with a ref#
    for idx, chunk in enumerate(chunks):
        chunked_docs.append({
            "filename": doc["filename"],
            "path": doc["path"],
            "chunk_id": idx,
            "chunk_content": chunk
        })

# Write the chunked doc to a new JSON file
with open("chunked_docs.json", "w") as f:
    json.dump(chunked_docs, f, indent=2)

print("Chunking complete. Cleaned and chunked data saved to chunked_docs.json")


########### or use LangChain
#  # I am only looking for Markdown (.md, .mdx) files
# valid_extensions = (".md", ".mdx")
# filtered_docs = [doc for doc in docs_data if doc["filename"].endswith(valid_extensions)]

# # sample of text splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,  # this could be adjusted
#     chunk_overlap=50,  # same here
#     length_function=len,
# )

# # Chunking the  docs
# chunked_docs = []
# for doc in filtered_docs:
#     chunks = text_splitter.split_text(doc["content"])
#     for i, chunk in enumerate(chunks):
#         chunked_docs.append({
#             "filename": doc["filename"],
#             "path": doc["path"],
#             "chunk_id": i,
#             "content": chunk
#         })
# #############--Optional--#############-
# # convert to pandas to view it
# with open("chunked_docs.json", "w") as f:
#     json.dump(chunked_docs, f, indent=2)
# chunked_df = pd.DataFrame(chunked_docs)
# chunked_df.head()
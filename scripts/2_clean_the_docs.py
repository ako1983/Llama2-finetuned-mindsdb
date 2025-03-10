import json
import re
import nltk

# Downloading NLTK 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# cleaning the data:
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text into words
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Optional, join nack the tokens
    cleaned = " ".join(tokens)
    return cleaned

# Loading JSON
with open("./docs/docs_data.json", "r") as f:
    docs = json.load(f)

cleaned_docs = []

# Iterate through each document in the JSON data
for doc in docs:
    # Extract content, filename, and path
    content = doc.get("content", "")
    filename = doc.get("filename", "unknown")
    path = doc.get("path", "unknown")
    
    # Clean the content
    cleaned_content = clean_text(content)
    
    # Append a new dictionary with cleaned data
    cleaned_docs.append({
        "filename": filename,
        "path": path,
        "cleaned_content": cleaned_content
    })

# Optionally, write the output
with open("cleaned_docs.json", "w") as f:
    json.dump(cleaned_docs, f, indent=2)

print("Data cleaning complete. Cleaned data saved to cleaned_docs.json")
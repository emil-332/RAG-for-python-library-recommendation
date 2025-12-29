import os
import json
import ollama
from tqdm import tqdm

# Folder containing your JSON files
DATA_DIR = "./enriched"

def get_usage_description(name, summary, readme_snippet):
    prompt = f"""
    You are a technical expert. Based on the library name and README snippet below, 
    write a short (2-sentence) summary of what a developer can achieve with this library.
    Use goal-oriented language like "This library is used to..."
    
    Library: {name}
    Summary: {summary}
    README Snippet: {readme_snippet[:800]}
    """

    try:
        response = ollama.chat(model='llama3.1', messages=[
            {'role': 'user', 'content': prompt},
        ])
        return response['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    # Loop through all files in enriched/
    for filename in tqdm(os.listdir(DATA_DIR)):
        if filename.endswith(".json"):
            path = os.path.join(DATA_DIR, filename)

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Skip ONLY if "usage_description" exists AND is not empty
            if data.get("usage_description") and data["usage_description"].strip():
                continue

            # Get text context from the first chunk
            first_chunk = data["chunks"][0]["text"] if data["chunks"] else ""

            # Generate the description
            description = get_usage_description(data["name"], data["summary"], first_chunk)
            data["usage_description"] = description

            # Save the updated JSON
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
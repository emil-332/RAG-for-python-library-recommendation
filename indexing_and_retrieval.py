import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_DIR = './enriched'         # Folder containing JSON files
INDEX_FILE = 'library_index.faiss'
METADATA_FILE = 'metadata.json'

# Load the SBERT model
model = SentenceTransformer(MODEL_NAME)

def build_vector_database():
    all_chunks = []
    metadata_map = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
                lib_data = json.load(f)

                # 1. Process standard chunks (README sections)
                for chunk_entry in lib_data.get('chunks', []):
                    # Access the .text key because chunk_entry is a dictionary
                    text_content = chunk_entry['text']
                    all_chunks.append(text_content)
                    metadata_map.append({
                        "name": lib_data['name'],
                        "tags": lib_data.get('tags', []),
                        "summary": lib_data.get('summary', ""),
                        "text": text_content,
                    })

                # 2. Process the LLM-enriched description (OUTSIDE the chunk loop)
                desc = lib_data.get('usage_description', "")
                if desc:
                    all_chunks.append(desc)
                    metadata_map.append({
                        "name": lib_data['name'],
                        "tags": lib_data.get('tags', []),
                        "summary": lib_data.get('summary', ""),
                        "text": desc,
                    })

    print(f"Encoding {len(all_chunks)} chunks from {len(os.listdir(DATA_DIR))} libraries...")

    # 3. Generate Embeddings
    # SBERT converts text into a fixed-size vector (384 dimensions for MiniLM)
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # 4. Create FAISS Index (using Inner Product for Cosine Similarity)
    # Normalize vectors first to use IndexFlatIP for cosine similarity
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # 5. Save index and metadata
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata_map, f)

    print("Database successfully built and saved locally.")

def retrieve_libraries(query, top_x=5):
    """Retrieves the top x matching chunks for a natural language query."""
    # Load index and metadata
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r') as f:
        metadata_map = json.load(f)

    # Encode and normalize user query
    query_embedding = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)

    # Search the index
    similarities, indices = index.search(query_embedding, top_x)

    results = []
    for i in range(top_x):
        idx = indices[0][i]
        results.append({
            "library": metadata_map[idx]['name'],
            "score": float(similarities[0][i]),
            "context_chunk": metadata_map[idx]['text'],
            "tags": metadata_map[idx]['tags']
        })

    return results

if __name__ == "__main__":
    # Uncomment to build the DB for the first time
    build_vector_database()

    user_goal = "I want to build a web scraper to collect news articles."
    matches = retrieve_libraries(user_goal, top_x=3)

    print(f"\nTop recommendations for: '{user_goal}'")
    for res in matches:
        print(f"- {res['library']} (Similarity: {res['score']:.4f})")
        print(f"  Tags: {res['tags']}")
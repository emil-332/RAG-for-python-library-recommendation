# chunk_readme.py

import json
from pathlib import Path

CLEAN_DIR = Path("clean")
CHUNKED_DIR = Path("chunked")

CHUNKED_DIR.mkdir(exist_ok=True)

MIN_WORDS = 200
MAX_WORDS = 500
NO_CHUNK_THRESHOLD = 800


def word_count(text: str) -> int:
    return len(text.split())


def split_paragraphs(text: str):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_text(text: str):
    words = word_count(text)

    if words <= NO_CHUNK_THRESHOLD:
        return [text]

    paragraphs = split_paragraphs(text)
    chunks = []

    current_chunk = []
    current_count = 0

    for para in paragraphs:
        para_words = word_count(para)

        if para_words > MAX_WORDS:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_count = 0
            chunks.append(para)
            continue

        if current_count + para_words <= MAX_WORDS:
            current_chunk.append(para)
            current_count += para_words
        else:
            if current_count >= MIN_WORDS:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_count = para_words
            else:
                current_chunk.append(para)
                current_count += para_words

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def main():
    for clean_file in CLEAN_DIR.glob("*.txt"):
        with open(clean_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        chunks = chunk_text(text)

        output = []
        for i, chunk in enumerate(chunks):
            output.append({
                "chunk_id": f"{clean_file.stem}::chunk_{i}",
                "text": chunk,
                "word_count": word_count(chunk),
            })

        out_file = CHUNKED_DIR / f"{clean_file.stem}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Chunked {clean_file.stem}: {len(output)} chunks")


if __name__ == "__main__":
    main()

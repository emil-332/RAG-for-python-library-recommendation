import json
from pathlib import Path

RAW_DIR = Path("raw")
CLEAN_DIR = Path("clean")
CHUNKED_DIR = Path("chunked")
ENRICHED_DIR = Path("enriched")

ENRICHED_DIR.mkdir(exist_ok=True)

TAG_KEYWORDS = {
    "web": ["http", "web", "api", "rest"],
    "data": ["data", "dataset", "csv", "json"],
    "ml": ["machine learning", "neural", "model"],
    "math": ["math", "algebra", "statistics"],
    "visualization": ["plot", "visualization", "chart"],
    "cli": ["command line", "cli", "terminal"],
    "ui": ["gui", "interface", "window"],
    "dev": ["test", "testing", "pytest", "lint"],
}

CLASSIFIER_MAP = {
    "artificial intelligence": "ml",
    "machine learning": "ml",
    "scientific": "math",
    "visualization": "visualization",
    "web": "web",
    "database": "data",
}


def tags_from_classifiers(classifiers):
    tags = set()
    for cls in classifiers:
        cls_lower = cls.lower()
        for key, tag in CLASSIFIER_MAP.items():
            if key in cls_lower:
                tags.add(tag)
    return tags


def tags_from_readme(text):
    tags = set()
    text = text.lower()

    for tag, keywords in TAG_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits >= 2:
            tags.add(tag)

    return tags


def main():
    for raw_file in RAW_DIR.glob("*.json"):
        name = raw_file.stem

        with open(raw_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        clean_file = CLEAN_DIR / f"{name}.txt"
        chunk_file = CHUNKED_DIR / f"{name}.json"

        if not clean_file.exists() or not chunk_file.exists():
            continue

        with open(clean_file, "r", encoding="utf-8") as f:
            clean_text = f.read()

        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        tags = set()
        tags |= tags_from_classifiers(raw.get("classifiers", []))
        tags |= tags_from_readme(clean_text)

        enriched = {
            "name": name,
            "summary": raw.get("summary"),
            "language": "python",
            "tags": sorted(tags),
            "chunks": chunks,
        }

        out_file = ENRICHED_DIR / f"{name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False)

        print(f"Tagged {name}: {sorted(tags)}")


if __name__ == "__main__":
    main()

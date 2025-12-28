import json
import time
from pathlib import Path
import requests

PYPI_URL = "https://pypi.org/pypi/{package}/json"

DATA_DIR = Path("data")
RAW_DIR = Path("raw")
PACKAGE_LIST = Path("package_list.txt")

RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_package_list():
    with open(PACKAGE_LIST, "r") as f:
        return [line.strip() for line in f if line.strip()]


def fetch_package_metadata(package_name: str) -> dict:
    url = PYPI_URL.format(package=package_name)
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def extract_relevant_fields(pypi_json: dict) -> dict:
    info = pypi_json.get("info", {})
    return {
        "name": info.get("name"),
        "summary": info.get("summary"),
        "long_description": info.get("description"),
        "description_content_type": info.get("description_content_type"),
        "classifiers": info.get("classifiers", []),
        "project_urls": info.get("project_urls", {}),
        "home_page": info.get("home_page"),
        "version": info.get("version"),
    }


def main():
    packages = load_package_list()

    for pkg in packages:
        print(f"Ingesting {pkg}...")
        try:
            raw = fetch_package_metadata(pkg)
            data = extract_relevant_fields(raw)

            out_file = RAW_DIR / f"{pkg}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            time.sleep(0.2)  # polite rate limiting

        except Exception as e:
            print(f"Failed to ingest {pkg}: {e}")


if __name__ == "__main__":
    main()

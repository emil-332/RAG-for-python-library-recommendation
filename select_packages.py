import requests
import time
from pathlib import Path
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed


PYPI_SIMPLE_URL = "https://pypi.org/simple/"
PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"
PYPSTATS_URL = "https://pypistats.org/api/packages/{package}/recent"

OUTPUT_FILE = Path("package_list.txt")

MIN_DOWNLOADS = 500_000
MIN_README_LENGTH = 200

MAX_WORKERS = 8
LOG_EVERY = 1_000

DS_CLASSIFIER_KEYWORDS = [
    "scientific",
    "engineering",
    "machine learning",
    "artificial intelligence",
    "statistics",
    "visualization",
    "data analysis",
    "numerical",
]

class SimpleIndexParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.packages = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for attr, value in attrs:
                if attr == "href":
                    pkg = value.strip("/").split("/")[-1]
                    self.packages.append(pkg)


def get_all_packages():
    r = requests.get(PYPI_SIMPLE_URL, timeout=30)
    r.raise_for_status()
    parser = SimpleIndexParser()
    parser.feed(r.text)
    return list(set(parser.packages))


def is_data_science_package(classifiers):
    for cls in classifiers:
        cls_lower = cls.lower()
        for kw in DS_CLASSIFIER_KEYWORDS:
            if kw in cls_lower:
                return True
    return False


def process_package(pkg: str):
    try:
        r = requests.get(PYPI_JSON_URL.format(package=pkg), timeout=10)
        r.raise_for_status()
        info = r.json().get("info", {})
    except Exception:
        return None

    classifiers = info.get("classifiers", [])
    if not classifiers:
        return None

    if not is_data_science_package(classifiers):
        return None

    try:
        r = requests.get(PYPSTATS_URL.format(package=pkg), timeout=10)
        r.raise_for_status()
        downloads = r.json()["data"]["last_month"]
    except Exception:
        return None

    if downloads < MIN_DOWNLOADS:
        return None

    desc = info.get("description", "")
    if not desc or len(desc) < MIN_README_LENGTH:
        return None

    return (pkg, downloads)


def main():
    print("Fetching package list from PyPI Simple Index...")
    packages = get_all_packages()
    total = len(packages)
    print(f"Total packages found: {total}")

    accepted = []
    processed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_package, pkg): pkg for pkg in packages}

        for future in as_completed(futures):
            processed += 1

            result = future.result()
            if result:
                accepted.append(result)
                print(f"Accepted: {result[0]} ({result[1]})")

            if processed % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (total - processed) / rate if rate > 0 else 0
                percent = (processed / total) * 100

                print(
                    f"Progress: {processed}/{total} ({percent:.2f}%) | "
                    f"Accepted: {len(accepted)} | "
                    f"ETA: {remaining/60:.1f} min"
                )

    accepted.sort(key=lambda x: x[1], reverse=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pkg, _ in accepted:
            f.write(pkg + "\n")

    elapsed_total = time.time() - start_time
    print(
        f"\nFinished full scan.\n"
        f"Accepted packages: {len(accepted)}\n"
        f"Total runtime: {elapsed_total/60:.1f} minutes\n"
        f"Output written to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()

import json
import re
from pathlib import Path

RAW_DIR = Path("raw")
CLEAN_DIR = Path("clean")

CLEAN_DIR.mkdir(exist_ok=True)

SECTION_BLACKLIST = [
    "installation",
    "install",
    "changelog",
    "contributing",
    "license",
    "authors",
    "credits",
    "tests",
    "release notes",
    "citation",
    "citing",
    "references",
    "acknowledgements",
    "contributors",
    "support",
    "getting help",
    "issues",
]

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAD6"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)

def remove_badges(text: str) -> str:
    text = re.sub(r"\[\!\[.*?\]\(.*?\)\]", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r".*badge.*", "", text, flags=re.IGNORECASE)
    return text


def remove_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def remove_code_blocks(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\n {4}.*", "", text)
    return text


def normalize_tables(text: str) -> str:
    lines = text.splitlines()
    output = []

    for line in lines:
        if line.strip().startswith("|") and "|" in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells:
                output.append(cells[0])
        else:
            output.append(line)

    return "\n".join(output)


def remove_link_lists(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    buffer = []
    link_lines = 0

    for line in lines:
        if line.strip().startswith("-") and "http" in line:
            buffer.append(line)
            link_lines += 1
        else:
            if buffer:
                if link_lines < 3:
                    cleaned.extend(buffer)
                buffer = []
                link_lines = 0
            cleaned.append(line)

    return "\n".join(cleaned)


def remove_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub("", text)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_sections(text: str):
    pattern = re.compile(r"\n#{1,6}\s+")
    parts = pattern.split("\n" + text)
    headers = pattern.findall("\n" + text)
    headers = [h.strip("# ").lower() for h in headers]

    if not headers:
        return [("body", text)]

    sections = []
    for header, content in zip(headers, parts[1:]):
        sections.append((header, content.strip()))

    return sections


def is_blacklisted_section(header: str) -> bool:
    return any(term in header for term in SECTION_BLACKLIST)


def remove_markdown_references(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(
        line for line in lines
        if not re.match(r"^\s*\[[^\]]+\]:\s*https?://", line)
    )


def remove_standalone_urls(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("http://") or stripped.startswith("https://"):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def remove_install_commands(text: str) -> str:
    patterns = [
        r"^\s*pip\s+install\s+.*$",
        r"^\s*conda\s+install\s+.*$",
        r"^\s*apt(-get)?\s+install\s+.*$",
        r"^\s*yum\s+install\s+.*$",
        r"^\s*brew\s+install\s+.*$",
    ]

    lines = []
    for line in text.splitlines():
        if any(re.match(p, line.strip(), re.IGNORECASE) for p in patterns):
            continue
        lines.append(line)

    return "\n".join(lines)


def clean_readme_text(text: str) -> str:
    text = remove_badges(text)
    text = remove_html(text)
    text = remove_code_blocks(text)

    text = normalize_tables(text)
    text = remove_link_lists(text)
    text = remove_emojis(text)

    text = remove_markdown_references(text)
    text = remove_standalone_urls(text)
    text = remove_install_commands(text)

    sections = split_sections(text)
    kept_sections = []

    for header, content in sections:
        if not is_blacklisted_section(header):
            kept_sections.append(content)

    text = "\n\n".join(kept_sections)

    return normalize_whitespace(text)


def main():
    for raw_file in RAW_DIR.glob("*.json"):
        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_text = data.get("long_description", "")
        if not raw_text.strip():
            continue

        cleaned = clean_readme_text(raw_text)

        if len(cleaned) < 200:
            continue

        out_file = CLEAN_DIR / f"{raw_file.stem}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"Cleaned {raw_file.stem}")


if __name__ == "__main__":
    main()

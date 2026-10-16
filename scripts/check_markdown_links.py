"""Validate local Markdown file links with the Python standard library."""

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'!?\[[^\]]+\]\(([^)\s]+)(?:\s+["\x27][^)]*["\x27])?\)')


def check_links(root: Path) -> list[str]:
    errors = []
    for document in sorted(root.rglob("*.md")):
        if any(part.startswith(".") for part in document.relative_to(root).parts[:-1]):
            continue
        for line_no, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            for match in LINK.finditer(line):
                url = match.group(1)
                if url.startswith(("#", "http://", "https://", "mailto:", "data:")):
                    continue
                path = unquote(url.split("#", 1)[0].split("?", 1)[0])
                if path and not (document.parent / path).exists():
                    errors.append(f"{document.relative_to(root)}:{line_no}: missing {path}")
    return errors


if __name__ == "__main__":
    issues = check_links(ROOT)
    if issues:
        raise SystemExit("\n".join(issues))
    print("Markdown relative links: OK")

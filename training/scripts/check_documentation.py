#!/usr/bin/env python3
"""Check maintained English documentation, relative links and experiment indexing."""

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
CJK = re.compile(r"[\u3400-\u9fff]")
# Preserve exact upstream attribution/work titles, not untranslated project prose.
PROVENANCE_NAMES = {
    "docs/japanese-additional-sources-20261009.md": ("桃治いの", "祭・fes-beat"),
    "docs/japanese-sources-20261009.md": ("前へ、すすむぞっ", "新しいあした", "新しい明日"),
}
LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")


def slug(text: str) -> str:
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]*>", "", text).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def body_lines(text: str) -> list[tuple[int, str]]:
    result, fence = [], None
    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = None
            continue
        if fence is None:
            result.append((number, line))
    return result


def anchors(text: str) -> set[str]:
    result, counts = set(), {}
    for _, line in body_lines(text):
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            base = slug(match[1])
            count = counts.get(base, 0)
            counts[base] = count + 1
            result.add(f"{base}-{count}" if count else base)
        result.update(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', line))
    return result


def check() -> int:
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.md"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    names = sorted(set(names))
    documents = {ROOT / name: (ROOT / name).read_text() for name in names}
    failures = []
    registry = documents[ROOT / "docs/failed-experiments.md"]
    for name in names:
        path, text = ROOT / name, documents[ROOT / name]
        for number, line in enumerate(text.splitlines(), 1):
            for literal in PROVENANCE_NAMES.get(name, ()):
                line = line.replace(literal, "")
            if CJK.search(line):
                failures.append(
                    f"{name}:{number}: non-English prose or unregistered provenance name"
                )
        for number, line in body_lines(text):
            for target in LINK.findall(line):
                target = target.strip("<>")
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc:
                    continue
                destination = (
                    (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
                )
                if not destination.exists():
                    failures.append(f"{name}:{number}: missing relative target {target}")
                elif parsed.fragment and destination.suffix == ".md":
                    content = documents.get(destination)
                    if content is None:
                        content = destination.read_text()
                    if unquote(parsed.fragment) not in anchors(content):
                        failures.append(f"{name}:{number}: missing heading {target}")
        title = text.splitlines()[0] if text else ""
        if (
            name.startswith("docs/")
            and re.match(r"# P\d+\b", title)
            and Path(name).name not in registry
        ):
            failures.append(
                f"{name}: P-series report missing from failed/unqualified experiment registry"
            )
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(
        f"Documentation checks passed: {len(names)} Markdown files; English, links and P-series index"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(check())

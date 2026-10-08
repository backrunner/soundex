# SPDX-License-Identifier: Apache-2.0
"""Read the work's own publisher fields, excluding recommendations and executable markup."""

from __future__ import annotations

import re
from html.parser import HTMLParser


class DefinitionFields(HTMLParser):
    """Collect definition-list text without treating related-work credits as authors."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fields: dict[str, list[str]] = {}
        self.current = ""
        self.text: list[str] = []
        self.key = ""
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.skip += 1
        if not self.skip and tag in {"dt", "dd"}:
            self.current, self.text = tag, []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.skip:
            self.skip -= 1
        if not self.skip and tag == self.current:
            value = " ".join("".join(self.text).split())
            if tag == "dt":
                self.key = value
            elif tag == "dd" and self.key:
                self.fields.setdefault(self.key, []).append(value)
            self.current, self.text = "", []

    def handle_data(self, data: str) -> None:
        if self.current and not self.skip:
            self.text.append(data)


def gongu_work_fields(source_html: str, expected_title: str) -> dict[str, str]:
    """Bind title and actual work owner, preserving composer/performance notes separately."""
    parser = DefinitionFields()
    parser.feed(source_html)
    fields = parser.fields
    if expected_title not in fields.get("저작물명", []):
        raise ValueError("publisher work title does not match the acquired recording")
    owners = fields.get("저작(권)자", [])
    if len(owners) != 1:
        raise ValueError("publisher work rights-owner field is missing or ambiguous")
    owner = re.sub(r"\s*\(저작물\s+\d+\s*건\)\s*$", "", owners[0]).strip()
    if not owner:
        raise ValueError("publisher work rights owner is empty")
    mapping = {
        "summary": "요약정보",
        "rights_statement": "이용조건",
        "work_kind": "저작물 속성",
        "collection_basis": "수집연계유형",
        "notes": "추가사항",
        "contributor": "기여자",
        "classification": "분류(장르)",
    }
    return {
        "rights_owner": owner,
        **{key: "\n".join(fields.get(label, [])) for key, label in mapping.items()},
    }

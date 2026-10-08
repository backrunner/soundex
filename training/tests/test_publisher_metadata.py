# SPDX-License-Identifier: Apache-2.0
"""Recommended tracks cannot supply the author of an acquired recording."""

import pytest

from data.publisher_metadata import gongu_work_fields


def test_own_work_fields_exclude_related_author_and_script_text() -> None:
    page = """<dl><dt>저작물명</dt><dd>Original</dd>
    <dt>저작(권)자</dt><dd><a>Actual composer</a> (저작물 2 건)</dd>
    <dt>추가사항</dt><dd>Original recording grant</dd></dl>
    <p class="author">Recommended artist</p>
    <script>document.write('<dt>저작(권)자</dt><dd>Fake</dd>')</script>"""
    fields = gongu_work_fields(page, "Original")
    assert fields["rights_owner"] == "Actual composer"
    assert fields["notes"] == "Original recording grant"
    with pytest.raises(ValueError, match="title"):
        gongu_work_fields(page, "Different work")


def test_missing_owner_cannot_be_filled_from_a_recommendation() -> None:
    page = '<dt>저작물명</dt><dd>Original</dd><p class="author">Other creator</p>'
    with pytest.raises(ValueError, match="rights-owner"):
        gongu_work_fields(page, "Original")

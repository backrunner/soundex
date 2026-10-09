# SPDX-License-Identifier: Apache-2.0
"""Listing legacy sources stays read-only; downloads require explicit selection."""

from pathlib import Path

import pytest

from scripts.download_datasets import main


def test_list_does_not_require_selection_or_create_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "untouched"
    monkeypatch.setattr("sys.argv", ["download_datasets", "--list", "--data-root", str(root)])
    assert main() == 0
    assert not root.exists()


def test_download_without_selection_fails_before_creating_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "untouched"
    monkeypatch.setattr("sys.argv", ["download_datasets", "--data-root", str(root)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not root.exists()

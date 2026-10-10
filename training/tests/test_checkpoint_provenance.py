"""A frozen archive must not inherit its enclosing live repository's HEAD."""

import hashlib
import json
from subprocess import CompletedProcess

import pytest

import checkpoint_state


def _archive(tmp_path, monkeypatch):
    repository = tmp_path / "run" / "source"
    (repository / "training").mkdir(parents=True)
    monkeypatch.setattr(
        checkpoint_state, "__file__", str(repository / "training/checkpoint_state.py")
    )
    monkeypatch.setattr(
        checkpoint_state.shutil, "which", lambda name: "git" if name == "git" else None
    )
    monkeypatch.setattr(checkpoint_state.platform, "platform", lambda: "test-platform")
    return repository


def test_frozen_archive_uses_receipt_instead_of_enclosing_git(tmp_path, monkeypatch):
    repository = _archive(tmp_path, monkeypatch)
    payload = json.dumps({"code_commit": "a" * 40}).encode()
    (repository.parent / "source-receipt.json").write_bytes(payload)

    def forbidden_git(*args, **kwargs):
        raise AssertionError("must not query enclosing live repository")

    monkeypatch.setattr(checkpoint_state.subprocess, "run", forbidden_git)
    result = checkpoint_state.capture_provenance()
    assert result["source_git_sha"] == "a" * 40
    assert result["source_git_sha_origin"] == "frozen-source-receipt"
    assert result["source_receipt_sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("commit", [None, "HEAD", "abc", "g" * 40])
def test_malformed_frozen_commit_is_rejected(tmp_path, monkeypatch, commit):
    repository = _archive(tmp_path, monkeypatch)
    (repository.parent / "source-receipt.json").write_text(json.dumps({"code_commit": commit}))
    with pytest.raises(ValueError, match="full code_commit"):
        checkpoint_state.capture_provenance()


def test_detached_worktree_uses_its_own_git_marker(tmp_path, monkeypatch):
    repository = _archive(tmp_path, monkeypatch)
    (repository / ".git").write_text("gitdir: /repository/.git/worktrees/frozen")
    (repository.parent / "source-receipt.json").write_text(json.dumps({"code_commit": "a" * 40}))

    def worktree_git(args, **kwargs):
        assert kwargs["cwd"] == repository
        return CompletedProcess(args, 0, stdout="b" * 40 + "\n", stderr="")

    monkeypatch.setattr(checkpoint_state.subprocess, "run", worktree_git)
    result = checkpoint_state.capture_provenance()
    assert result["source_git_sha"] == "b" * 40
    assert result["source_git_sha_origin"] == "git"


def test_archive_without_receipt_does_not_claim_live_git_version(tmp_path, monkeypatch):
    _archive(tmp_path, monkeypatch)

    def forbidden_git(*args, **kwargs):
        raise AssertionError("must not query enclosing live repository")

    monkeypatch.setattr(checkpoint_state.subprocess, "run", forbidden_git)
    result = checkpoint_state.capture_provenance()
    assert result["source_git_sha"] is None
    assert result["source_git_sha_origin"] == "unversioned-source-archive"

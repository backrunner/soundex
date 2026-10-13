# SPDX-License-Identifier: Apache-2.0
"""Exercise the container shell against isolated check/train stubs, without Docker."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def entrypoint(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    workspace = tmp_path / "workspace"
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "configs").mkdir()
    (workspace / "configs/default.yaml").write_text("{}\n")
    (workspace / "scripts/check_env.py").write_text(
        "import os, sys\nsys.exit(int(os.environ.get('CHECK_STATUS', '0')))\n"
    )
    (workspace / "train.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path('invocation.json').write_text(json.dumps({'args': sys.argv[1:], "
        "'music': os.environ.get('MUSIC_LIBRARY_PATH'), "
        "'speech': os.environ.get('SPEECH_LIBRARY_PATH')}))\n"
    )
    source = Path(__file__).parents[1] / "scripts/entrypoint.sh"
    script = workspace / "entrypoint.sh"
    script.write_text(source.read_text().replace("cd /workspace", f"cd '{workspace}'"))
    executable = tmp_path / "bin"
    executable.mkdir()
    (executable / "python").symlink_to(sys.executable)
    env = {"PATH": f"{executable}:/usr/bin:/bin", "HOME": str(tmp_path)}
    return script, env


def test_prepared_paths_and_explicit_manifest_reach_training(
    entrypoint: tuple[Path, dict[str, str]],
) -> None:
    script, env = entrypoint
    manifest = script.parent / "explicit paths.yaml"
    manifest.write_text("training_data: {}\n")
    subprocess.run(
        ["bash", str(script)],
        env={
            **env,
            "MUSIC_LIBRARY_PATH": "/data/music",
            "SPEECH_LIBRARY_PATH": "/data/speech",
            "PATHS_MANIFEST": str(manifest),
            "EXTRA_TRAIN_ARGS": "--resume /data/resume.pth",
        },
        check=True,
    )
    call = json.loads((script.parent / "invocation.json").read_text())
    assert call == {
        "args": [
            "--config",
            "configs/default.yaml",
            "--paths-manifest",
            str(manifest),
            "--resume",
            "/data/resume.pth",
        ],
        "music": "/data/music",
        "speech": "/data/speech",
    }


@pytest.mark.parametrize(
    ("overrides", "code", "message"),
    [
        ({"DOWNLOAD_DATASETS": "slakh2100"}, 2, "acquisition was removed"),
        ({"PREPROCESS": "1"}, 2, "acquisition was removed"),
        ({"PATHS_MANIFEST": "/missing/manifest.yaml"}, 1, "Paths manifest not found"),
        ({"CHECK_STATUS": "7"}, 7, ""),
        ({"CHECK_ENV_ONLY": "1"}, 0, ""),
    ],
)
def test_no_training_on_invalid_or_check_only_request(
    entrypoint: tuple[Path, dict[str, str]],
    overrides: dict[str, str],
    code: int,
    message: str,
) -> None:
    script, env = entrypoint
    result = subprocess.run(
        ["bash", str(script)], env={**env, **overrides}, capture_output=True, text=True
    )
    assert result.returncode == code
    assert message in result.stderr
    assert not (script.parent / "invocation.json").exists()

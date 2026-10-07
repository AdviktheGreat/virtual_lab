"""Code in a session calling Proto's tools as a library, with the interpreter Proto is installed in.

Skipped unless Proto is installed: with uv tool install "proto-tools[mcp] @ git+https://github.com/evo-design/proto-tools.git",
or in the interpreter PROTO_TOOLS_PYTHON names. Nothing is run on Proto's servers or on Modal.
"""

import os
from pathlib import Path

import pytest

from virtual_lab import LocalSession
from virtual_lab.resources import check_installed

PROTO_PYTHON = Path(
    os.environ.get("PROTO_TOOLS_PYTHON") or Path.home() / ".local/share/uv/tools/proto-tools/bin/python"
)

pytestmark = pytest.mark.skipif(not PROTO_PYTHON.exists(), reason=f"Proto is not installed at {PROTO_PYTHON}")


def test_code_in_a_session_imports_protos_tools_and_has_what_running_them_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROTO_API_KEY", "not-a-real-key")
    with LocalSession(
        tmp_path,
        python=str(PROTO_PYTHON),
        warn=False,
        forward_env=("PROTO_API_KEY",),
        software={"proto_tools": "Proto's tools for protein and sequence design."},
    ) as session:
        assert check_installed(session, ["proto_tools"], [], what="the software added to it") == (set(), {})
        result = session.run(
            "import os, proto_tools\n"
            "from proto_tools import ESMFoldConfig, run_esmfold\n"
            "print(sum(name.startswith('run_') for name in dir(proto_tools)))\n"
            "print(ESMFoldConfig(device='proto').device, callable(run_esmfold))\n"
            "print(os.environ['PROTO_API_KEY'], os.environ['HOME'])\n"
        )

    assert result.status == "ok", result.error
    count, config, environment = result.output.strip().splitlines()
    assert int(count) >= 100
    assert config == "proto True"
    # The key reaches code that runs on Proto's servers, and HOME finds Modal's token, in ~/.modal.toml
    assert environment == f"not-a-real-key {Path.home()}"

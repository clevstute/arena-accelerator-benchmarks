"""Backport the upstream same-device import check to the Arena TorchNPU 2.6.0 venv."""

from importlib.metadata import version
from pathlib import Path
from shutil import copy2
from stat import S_IMODE
import sys
import sysconfig
from tempfile import TemporaryDirectory


def patch_import_guard(path):
    old = 'if acc.type != "cpu":'
    new = 'if acc.type not in ("cpu", "npu"):'
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    if lines.count(new) == 1 and old not in lines:
        return
    if lines.count(old) != 1 or new in lines:
        raise RuntimeError(f"Unexpected TorchNPU import guard in {path}.")

    updated = source.replace(f"\n{old}\n", f"\n{new}\n", 1)
    if updated == source:
        raise RuntimeError(f"Could not locate the TorchNPU import guard in {path}.")
    compile(updated, str(path), "exec")
    backup = path.with_suffix(".py.before-arena-npu-guard")
    if not backup.exists():
        copy2(path, backup)
    with TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / path.name
        staged.write_text(updated, encoding="utf-8")
        staged.chmod(S_IMODE(path.stat().st_mode))
        staged.replace(path)


if __name__ == "__main__":
    venv = Path("/opt/arena-venv").resolve()
    installed = version("torch-npu")
    path = (Path(sysconfig.get_path("purelib")) / "torch_npu/__init__.py").resolve()
    if Path(sys.prefix).resolve() != venv or not path.is_relative_to(venv):
        raise SystemExit("Run with the dedicated /opt/arena-venv/bin/python.")
    if installed != "2.6.0":
        raise SystemExit(f"Expected TorchNPU 2.6.0, found {installed}.")

    patch_import_guard(path)
    print(f"TORCH_NPU_IMPORT_GUARD_OK {path}")

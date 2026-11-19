"""Paths for this editable simulation workspace, independent of shell cwd."""
from pathlib import Path
import re

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = WORKSPACE_ROOT / "assets"
CONFIG_DIR = WORKSPACE_ROOT / "config"
EXAMPLES_DIR = WORKSPACE_ROOT / "examples"
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
DATA_DIR = WORKSPACE_ROOT / "datasets"
CHECKS_DIR = OUTPUTS_DIR / "_checks"
TEST_DIR = OUTPUTS_DIR / "test"


def relocated_evaluation_path(value: str | Path) -> Path:
    """Find a run or file referenced by an older flat evaluation directory."""
    path = Path(value).expanduser()
    absolute = path.absolute()
    if absolute.is_relative_to(TEST_DIR):
        parts = absolute.relative_to(TEST_DIR).parts
        if parts:
            match = re.fullmatch(r"(.+_rgbd?)_(ind_ood|ind|ood)_(\d{8}_\d{6})", parts[0])
            if match and not (TEST_DIR / parts[0]).exists():
                new_run = TEST_DIR.joinpath(*match.groups())
                if new_run.is_dir():
                    return new_run.joinpath(*parts[1:])
    return path


def relocated_checkpoint_path(value: str | Path) -> Path:
    """Locate a checkpoint directory or file named in a pre-migration run."""
    path = Path(value).expanduser()
    train = OUTPUTS_DIR / "train"
    absolute = path.absolute()
    if absolute.is_relative_to(train):
        parts = absolute.relative_to(train).parts
        if parts[1:3] == ("artifacts", "checkpoints"):
            run = train / parts[0]
            if not (run / "artifacts/checkpoints").exists():
                return (run / "checkpoints").joinpath(*parts[3:])
    return path


def checkpoint_dataset_path(value: str | Path) -> Path:
    """Resolve a checkpoint's dataset root, including the pre-move location."""
    path = (WORKSPACE_ROOT / Path(value).expanduser()).resolve()
    old_root = OUTPUTS_DIR / "datasets"
    if path.is_relative_to(old_root):
        return (DATA_DIR / path.relative_to(old_root)).resolve()
    return path

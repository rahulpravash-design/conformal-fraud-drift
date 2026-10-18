from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_configs(config_dir: str | Path = "configs") -> dict:
    d = Path(config_dir)
    return {
        "data": load_yaml(d / "data.yaml"),
        "experiment": load_yaml(d / "experiment.yaml"),
        "methods": load_yaml(d / "methods.yaml"),
    }

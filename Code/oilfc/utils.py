"""Config loading, reproducibility and path helpers."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

# Code/ directory (parent of this package). All relative paths in config.yaml
# are resolved against it, so scripts work no matter where you launch them.
CODE_DIR = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = CODE_DIR / "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def override(cfg: dict, **kwargs) -> dict:
    """Return a deep copy of cfg with dotted-key overrides, e.g. override(cfg, **{"model.hidden_size": 128})."""
    new = copy.deepcopy(cfg)
    for dotted, value in kwargs.items():
        node = new
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return new


def path(cfg: dict, key: str, *parts: str) -> Path:
    """Absolute path for cfg['paths'][key] / parts, created if needed."""
    p = CODE_DIR / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p.joinpath(*parts) if parts else p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(prefer_gpu: bool = True):
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(obj, file: str | Path) -> None:
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(file: str | Path):
    with open(file) as f:
        return json.load(f)


def import_script(filename: str):
    """Import a numbered script such as '04_train_rnn.py' (not a valid module name) from another script."""
    import importlib.util

    file = CODE_DIR / filename
    spec = importlib.util.spec_from_file_location("script_" + file.stem, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_name(model: str, features: str, horizon: int, seed: int, tag: str = "") -> str:
    """Canonical file stem for one training run, e.g. 'lstm_all_h5_s0'."""
    name = f"{model}_{features}_h{horizon}_s{seed}"
    return f"{name}_{tag}" if tag else name

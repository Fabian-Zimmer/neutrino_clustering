"""Run configs: one YAML file per simulation run (see configs/).

Every script in scripts/ takes a config path as its only argument and reads the
section it needs. Relative paths (run_dir, data_dir) are relative to the repo
root; $VARIABLES and ~ are expanded, so cluster paths like $HOME/... work too.
"""

import argparse
import os
from pathlib import Path

import yaml

# Repo root (the package is installed in editable mode by `uv sync`).
ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    """Absolute path; relative paths are taken relative to the repo root."""
    p = Path(os.path.expandvars(str(path))).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_config(path):
    """Load a run config and make its directory entries absolute."""
    with open(path) as file:
        cfg = yaml.safe_load(file)

    for key, value in cfg.items():
        if key.endswith(("_dir", "_directory")) and value is not None:
            cfg[key] = str(resolve(value))

    return cfg


def config_from_cli(description):
    """Parse `script.py <config.yaml>` and return the loaded config."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="path to a run config, e.g. configs/dm_halos.yaml")
    args = parser.parse_args()
    return load_config(args.config)

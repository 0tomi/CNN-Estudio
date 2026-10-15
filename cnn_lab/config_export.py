"""Reusable editor JSON: architecture and hyperparameters, never model weights."""
import json
from .model import Config


def config_json(config):
    canonical=Config.from_dict(config) if isinstance(config,dict) else config
    return json.dumps(canonical.dict(),ensure_ascii=False,indent=2,allow_nan=False)

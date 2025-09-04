"""Model registry loader for evaluation harness.

Reads ``configs/models.yaml`` and returns configuration for a
specific ``model_id``.  Registry entries may specify whether chat
templates should be applied and provide generation defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULTS: Dict[str, Any] = {
    "use_chat_template": False,
    "system_prompt": "Answer with the exact shortest span from the prompt. No explanations.",
    "eos_token_id": None,
    "pad_token_id": None,
    "max_new_tokens": 32,
}


def load_model_config(model_id: str, path: str | Path = "configs/models.yaml") -> Dict[str, Any]:
    """Return registry settings for ``model_id``.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier used as key in ``models.yaml``.
    path:
        Location of the registry file relative to the repository root.
    """

    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    defaults = {**_DEFAULTS, **data.get("defaults", {})}
    model_cfg = data.get(model_id, {})
    merged = {**defaults, **model_cfg}
    return merged

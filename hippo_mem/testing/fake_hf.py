"""Deterministic fake Transformers backend for tests.

This module monkeypatches :mod:`transformers` loaders so that calls to
:func:`~transformers.AutoModelForCausalLM.from_pretrained` and
:func:`~transformers.AutoTokenizer.from_pretrained` return tiny, fully
in-memory fixtures when the caller requests :data:`FAKE_MODEL_ID` or the
legacy ``models/tiny-gpt2`` path used throughout the tests.  The fake
artifacts use a GPT-2 configuration with a minimal vocabulary and
produce deterministic weights so adapter and consolidation tests can run
without downloading or storing actual checkpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizer,
)

FAKE_MODEL_ID = "hippo/fake-tiny-gpt2"
_FAKE_PATH_SUFFIX = "models/tiny-gpt2"


def _normalize_model_id(value: object | None) -> str:
    """Return a normalised string representation for ``value``."""

    if value is None:
        return ""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    text = str(value).strip()
    if not text:
        return ""
    return text.replace("\\", "/").rstrip("/")


def is_fake_model_id(value: object | None) -> bool:
    """Return ``True`` when ``value`` refers to the fake HF model."""

    norm = _normalize_model_id(value)
    if not norm:
        return False
    if norm == FAKE_MODEL_ID:
        return True
    if norm == _FAKE_PATH_SUFFIX:
        return True

    parts = tuple(part for part in PurePosixPath(norm).parts if part not in {"", ".", "..", "/"})
    if len(parts) >= 2:
        if tuple(parts[-2:]) == tuple(FAKE_MODEL_ID.split("/")):
            return True
        if tuple(parts[-2:]) == tuple(_FAKE_PATH_SUFFIX.split("/")):
            return True
    return False


def resolve_fake_model_id(value: object | None) -> str:
    """Return ``FAKE_MODEL_ID`` when ``value`` denotes the fake backend."""

    norm = _normalize_model_id(value)
    if not norm:
        return ""
    return FAKE_MODEL_ID if is_fake_model_id(norm) else norm


_FAKE_VOCAB: Sequence[str] = (
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "hippo",
    "memory",
    "module",
    "retrieval",
    "episodic",
    "semantic",
    "spatial",
    "adapter",
    "gate",
    "write",
    "read",
    "done",
)


class FakeTokenizer(PreTrainedTokenizer):
    """Whitespace tokenizer backed by a tiny static vocabulary."""

    vocab_files_names = {"vocab_file": "vocab.json"}

    def __init__(self) -> None:
        # why: ``PreTrainedTokenizer`` consults ``get_vocab`` inside ``__init__``.  The
        # lookup needs the backing maps to exist beforehand, so populate them before
        # delegating to the parent constructor.
        self._token_to_id = {tok: idx for idx, tok in enumerate(_FAKE_VOCAB)}
        self._id_to_token = {idx: tok for tok, idx in self._token_to_id.items()}

        super().__init__(
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
        )

        self.model_input_names = ["input_ids", "attention_mask"]
        self.padding_side = "right"
        self.truncation_side = "right"
        self.model_max_length = 64

    @property
    def vocab_size(self) -> int:  # type: ignore[override]
        return len(self._token_to_id)

    def get_vocab(self) -> dict[str, int]:  # type: ignore[override]
        return dict(self._token_to_id)

    def _tokenize(self, text: str) -> List[str]:  # type: ignore[override]
        tokens: List[str] = []
        for raw in text.strip().split():
            token = raw.lower()
            tokens.append(token if token in self._token_to_id else self.unk_token)
        return tokens

    def _convert_token_to_id(self, token: str) -> int:  # type: ignore[override]
        return self._token_to_id.get(token, self._token_to_id[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:  # type: ignore[override]
        return self._id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: Iterable[str]) -> str:  # type: ignore[override]
        filtered = [
            tok for tok in tokens if tok not in {self.pad_token, self.bos_token, self.eos_token}
        ]
        return " ".join(filtered)

    def build_inputs_with_special_tokens(  # type: ignore[override]
        self,
        token_ids_0: List[int],
        token_ids_1: List[int] | None = None,
    ) -> List[int]:
        tokens = [self.bos_token_id] + list(token_ids_0)
        if token_ids_1:
            tokens += list(token_ids_1)
        tokens.append(self.eos_token_id)
        return tokens

    def get_special_tokens_mask(  # type: ignore[override]
        self,
        token_ids_0: List[int],
        token_ids_1: List[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, True)
        mask = [0] * len(token_ids_0)
        if token_ids_1:
            mask += [0] * len(token_ids_1)
        return [1] + mask + [1]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:  # type: ignore[override]
        return 3 if pair else 2

    def save_vocabulary(  # type: ignore[override]
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        vocab_path = path / f"{prefix}vocab.json"
        with vocab_path.open("w", encoding="utf-8") as handle:
            json.dump(self._token_to_id, handle, indent=2, ensure_ascii=False)
        return (str(vocab_path),)


def _build_fake_config(config: GPT2Config | None = None) -> GPT2Config:
    """Return a GPT-2 config aligned with the fake tokenizer."""

    if config is not None:
        cfg = config.__class__.from_dict(config.to_dict())
    else:
        cfg = GPT2Config(
            vocab_size=len(_FAKE_VOCAB),
            n_positions=64,
            n_ctx=64,
            n_embd=64,
            n_layer=2,
            n_head=4,
        )
    cfg.vocab_size = len(_FAKE_VOCAB)
    cfg.bos_token_id = _FAKE_VOCAB.index("<bos>")
    cfg.eos_token_id = _FAKE_VOCAB.index("<eos>")
    cfg.pad_token_id = _FAKE_VOCAB.index("<pad>")
    return cfg


def _build_fake_model(config: GPT2Config | None = None) -> GPT2LMHeadModel:
    cfg = _build_fake_config(config)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg)
    model.config.pad_token_id = cfg.pad_token_id
    model.config.eos_token_id = cfg.eos_token_id
    model.config.bos_token_id = cfg.bos_token_id
    return model


def _build_fake_tokenizer(**kwargs) -> FakeTokenizer:
    tokenizer = FakeTokenizer()
    padding_side = kwargs.get("padding_side")
    if padding_side:
        tokenizer.padding_side = padding_side
    truncation_side = kwargs.get("truncation_side")
    if truncation_side:
        tokenizer.truncation_side = truncation_side
    return tokenizer


_PATCHED = False
_ORIG_MODEL_FROM_PRETRAINED = AutoModelForCausalLM.from_pretrained.__func__
_ORIG_TOKENIZER_FROM_PRETRAINED = AutoTokenizer.from_pretrained.__func__


def _patched_model_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    if not is_fake_model_id(pretrained_model_name_or_path):
        return _ORIG_MODEL_FROM_PRETRAINED(
            cls, pretrained_model_name_or_path, *model_args, **kwargs
        )
    config = kwargs.pop("config", None)
    state_dict = kwargs.pop("state_dict", None)
    torch_dtype = kwargs.get("torch_dtype")
    model = _build_fake_model(config)
    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            # why: maintain parity with transformers by not failing when keys differ
            pass
    if torch_dtype is not None:
        model.to(dtype=torch_dtype)
    return model


def _patched_tokenizer_from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
    if is_fake_model_id(pretrained_model_name_or_path):
        return _build_fake_tokenizer(**kwargs)

    maybe_path = pretrained_model_name_or_path
    if isinstance(maybe_path, (str, os.PathLike)):
        path = Path(maybe_path)
        config_path = path / "tokenizer_config.json"
        if path.exists() and path.is_dir() and config_path.exists():
            try:
                data = json.loads(config_path.read_text())
            except Exception:
                data = {}
            tok_class = data.get("tokenizer_class") or data.get("tokenizer_class_python")
            if tok_class == "FakeTokenizer":
                return _build_fake_tokenizer(**kwargs)

    return _ORIG_TOKENIZER_FROM_PRETRAINED(cls, pretrained_model_name_or_path, *inputs, **kwargs)


def _patch_transformers() -> None:
    global _PATCHED
    if _PATCHED:
        return
    AutoModelForCausalLM.from_pretrained = classmethod(_patched_model_from_pretrained)
    AutoTokenizer.from_pretrained = classmethod(_patched_tokenizer_from_pretrained)
    _PATCHED = True


_patch_transformers()

__all__ = [
    "FAKE_MODEL_ID",
    "FakeTokenizer",
    "is_fake_model_id",
    "resolve_fake_model_id",
]

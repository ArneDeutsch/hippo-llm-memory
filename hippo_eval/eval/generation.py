"""Token generation helpers for the evaluation harness."""

from __future__ import annotations

import re

from hippo_eval.metrics.scoring import (
    em_norm,
    em_raw,
    enforce_short_answer,
    enforce_udlr,
    f1,
)

from .encode import encode_prompt

FORMAT_VIOL_RE = re.compile(r"\n|\.$")


def apply_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Return ``user_prompt`` with the tokenizer's chat template applied."""

    has_chat = hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    )
    if not has_chat:
        return user_prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    *,
    long_context: bool = False,
) -> tuple[str, int, int]:
    """Generate ``max_new_tokens`` from ``model`` for ``prompt``."""

    if long_context:
        prompt = f"{prompt} [CTX]"
    inputs = encode_prompt(
        tokenizer,
        prompt,
        model.device,
        use_chat_template=False,
        system_prompt=None,
    )
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    in_tok = int(inputs["input_ids"].shape[-1])
    gen = out[:, in_tok:]
    gen_tok = int(gen.shape[-1])
    raw_pred = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
    return raw_pred, in_tok, gen_tok


def postprocess(
    raw_pred: str,
    task,
    suite: str | None,
    compute_metrics: bool,
) -> tuple[str, float | None, float | None, float | None, int, int, int, int]:
    """Normalize ``raw_pred`` and compute metrics for ``task``."""

    suite_is_spatial = suite in {"spatial", "spatial_multi"}
    if suite_is_spatial:
        pred = enforce_udlr(raw_pred)
        fmt = int(pred != raw_pred.strip().upper())
        pred_len = len(pred)
        gold_len = len(task.answer)
        overlong = int(pred_len > gold_len)
        em_r = em_raw(pred, task.answer) if compute_metrics else None
        em_n = em_norm(pred, task.answer) if compute_metrics else None
        f1_val = 1.0 if pred == task.answer else 0.0 if compute_metrics else None
    else:
        pred = enforce_short_answer(raw_pred)
        pred_len = len(pred.split())
        gold_len = len(task.answer.split())
        overlong = int(pred_len > gold_len)
        fmt = int(bool(FORMAT_VIOL_RE.search(raw_pred)) or pred != raw_pred)
        em_r = em_raw(pred, task.answer) if compute_metrics else None
        em_n = em_norm(pred, task.answer) if compute_metrics else None
        f1_val = f1(pred, task.answer) if compute_metrics else None
    return pred, em_r, em_n, f1_val, overlong, fmt, pred_len, gold_len


__all__ = ["apply_chat_template", "generate", "postprocess"]

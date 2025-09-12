# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class Badge:
    """Status badge with optional target link."""

    name: str
    ok: Optional[bool]
    href: str | None = None

    def color(self) -> str:
        if self.ok is True:
            return "brightgreen"
        if self.ok is False:
            return "red"
        return "yellow"

    def markdown(self) -> str:
        color = self.color()
        href = self.href or "#"
        return f"[![{self.name}-{color}](https://img.shields.io/badge/{self.name}-{color})]({href})"


def render_panel(badges: Iterable[Badge]) -> str:
    """Return badges as a Markdown string sorted by health."""

    order = {False: 0, None: 1, True: 2}
    ordered = sorted(badges, key=lambda b: order[b.ok])
    return " ".join(b.markdown() for b in ordered)

"""Shared baseline specification type."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    name: str
    year: int
    paper_role: str
    args: tuple[str, ...]

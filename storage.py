#!/usr/bin/env python3
"""Lightweight persistent storage for training runs."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


@dataclass
class RunStore:
    """
    Minimal run storage helper.

    Creates a dedicated directory under ``runs/`` (by default) with:
    - ``meta.json`` capturing run configuration and creation time.
    - ``metrics.jsonl`` append-only log for per-step metrics.
    - ``checkpoints/*.pt`` for saved weights.
    """

    run_type: str
    root: Path = Path("runs")
    name: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ts = _timestamp()
        self.name = self.name or f"{self.run_type}_{ts}"
        self.dir = (self.root / self.name).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)

        meta = {"run_type": self.run_type, "created_utc": ts, "config": self.config}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        self.metrics_path = self.dir / "metrics.jsonl"
        self._metrics_fh = self.metrics_path.open("a", encoding="utf-8")

        self.ckpt_dir = self.dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        """Append a metrics record."""
        if not self._metrics_fh:
            return
        rec = {"ts": time.time(), **record}
        self._metrics_fh.write(json.dumps(rec) + "\n")
        self._metrics_fh.flush()

    def save_checkpoint(self, state_dict: Dict[str, Any], name: str = "policy.pt") -> Path:
        path = (self.ckpt_dir / name).resolve()
        torch.save(state_dict, path)
        return path

    def close(self) -> None:
        if self._metrics_fh:
            self._metrics_fh.close()
            self._metrics_fh = None

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

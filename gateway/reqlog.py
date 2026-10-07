"""Two-tier request logging.

meta log  — always on when configured: identity, route, status, timing, token counts. No prompt text.
body log  — opt-in, separate file: prompt and completion bodies. Keep its retention short.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class RequestLogger:
    def __init__(self, meta_path: str | None, body_path: str | None):
        self._meta = self._open(meta_path)
        self._body = self._open(body_path)
        self._lock = threading.Lock()

    @staticmethod
    def _open(path: str | None):
        if not path:
            return None
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return open(p, "a", encoding="utf-8")

    @property
    def body_enabled(self) -> bool:
        return self._body is not None

    def meta(self, record: dict) -> None:
        self._write(self._meta, record)

    def body(self, record: dict) -> None:
        self._write(self._body, record)

    def _write(self, fh, record: dict) -> None:
        if fh is None:
            return
        record.setdefault("ts", time.time())
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            fh.write(line + "\n")
            fh.flush()

    def close(self) -> None:
        for fh in (self._meta, self._body):
            if fh is not None:
                fh.close()

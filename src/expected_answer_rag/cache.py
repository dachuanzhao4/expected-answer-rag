from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class JsonCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: Dict[str, Any] = {}
        self._lock = threading.RLock()
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.data[key] = value
            self.flush()

    def flush(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(self.path)

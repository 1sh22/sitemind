from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from sqlitedict import SqliteDict


class ProjectStore:
    """Simple SQLite-backed key-value store for projects."""

    def __init__(self, db_path: str = "app/data/projects.sqlite") -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path

    def save_project(self, data: Dict[str, Any]) -> str:
        project_id = data.get("project_id")
        if not project_id:
            project_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            data["project_id"] = project_id
        with SqliteDict(self.db_path) as db:
            db[project_id] = data
            db.commit()
        return project_id

    def load_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        with SqliteDict(self.db_path) as db:
            return db.get(project_id)

    def export_json(self, data: Dict[str, Any], out_dir: str = "app/data") -> str:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"project_{data.get('project_id','unknown')}.json"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def list_projects(self, limit: int = 10) -> list[tuple[str, str]]:
        """Return [(project_id, created_at_iso)] for recent projects."""
        items: list[tuple[str, str]] = []
        with SqliteDict(self.db_path) as db:
            for key in db.keys():
                try:
                    created = db[key].get("created_at") or ""
                except Exception:
                    created = ""
                items.append((str(key), str(created)))
        items.sort(key=lambda x: x[0], reverse=True)
        return items[:limit]



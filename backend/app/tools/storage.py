import json
from typing import Any

from app.config import DATA_DIR


def read_store(filename: str, key: str) -> list[dict[str, Any]]:
    path = DATA_DIR / filename
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object in data file")
        rows = data.get(key)
        if not isinstance(rows, list):
            raise ValueError("Expected a list in data file")
        return rows
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"The hospital {key} data file is invalid.") from exc


def write_store(filename: str, key: str, rows: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps({key: rows}, indent=2), encoding="utf-8")
    temp_path.replace(path)


def next_id(prefix: str, rows: list[dict[str, Any]], field: str) -> str:
    numbers = []
    for row in rows:
        value = str(row.get(field, ""))
        suffix = value.removeprefix(prefix)
        if value.startswith(prefix) and suffix.isdigit():
            numbers.append(int(suffix))
    return f"{prefix}{max(numbers, default=0) + 1:03d}"

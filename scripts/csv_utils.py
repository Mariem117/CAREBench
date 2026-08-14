"""Small CSV helpers shared by scoring scripts."""

from __future__ import annotations

import csv
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], preferred: list[str] | None = None) -> None:
    if not rows:
        return
    keys: list[str] = []
    for key in preferred or []:
        if any(key in row for row in rows) and key not in keys:
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def upsert_rows(
    existing: list[dict],
    new_rows: list[dict],
    key_cols: tuple[str, ...],
) -> list[dict]:
    index = {
        tuple(str(row.get(col, "")) for col in key_cols): idx
        for idx, row in enumerate(existing)
    }
    output = list(existing)
    for row in new_rows:
        key = tuple(str(row.get(col, "")) for col in key_cols)
        if key in index:
            output[index[key]] = {**output[index[key]], **row}
        else:
            index[key] = len(output)
            output.append(row)
    return output

"""Read a dataset into SQLite and describe it for the LLM."""
import json
import os
import re
import secrets
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTED = {".csv", ".tsv", ".xlsx", ".xls", ".json"}
DATASET_ID = re.compile(r"^[0-9a-f]{32}$")


class DatasetNotFound(LookupError):
    pass


def qi(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def db_path(dataset_id: str) -> Path:
    path = DATA_DIR / f"{dataset_id}.db"
    if not DATASET_ID.fullmatch(dataset_id or "") or not path.exists():
        raise DatasetNotFound("Unknown dataset id")
    return path


def _read_file(filename: str, raw: bytes) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type. Use: {', '.join(sorted(SUPPORTED))}")
    if ext == ".json":
        return pd.DataFrame(json.loads(raw.decode("utf-8-sig")))
    temp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        temp.write(raw)
        temp.close()
        if ext in {".xlsx", ".xls"}:
            sheets = pd.read_excel(temp.name, sheet_name=None)
            if not sheets:
                raise ValueError("The file contains no data")
            return next(iter(sheets.values()))
        for encoding in ("utf-8-sig", "cp1256", "latin-1"):
            try:
                return pd.read_csv(
                    temp.name,
                    encoding=encoding,
                    sep="\t" if ext == ".tsv" else None,
                    engine="python",
                )
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode the file")
    finally:
        os.unlink(temp.name)


def _clean_columns(columns) -> list[str]:
    result = []
    used = set()
    for index, column in enumerate(columns, 1):
        name = re.sub(r"\s+", " ", str(column)).strip() or f"column_{index}"
        original = name
        suffix = 2
        while name.casefold() in used:
            name = f"{original}_{suffix}"
            suffix += 1
        used.add(name.casefold())
        result.append(name)
    return result


def create_dataset(filename: str, raw: bytes) -> dict:
    try:
        frame = _read_file(filename, raw).dropna(how="all").dropna(axis=1, how="all")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read '{filename}': {exc}") from exc
    if frame.empty:
        raise ValueError("The file contains no data")

    frame.columns = _clean_columns(frame.columns)
    dataset_id = secrets.token_hex(16)
    path = DATA_DIR / f"{dataset_id}.db"
    table = re.sub(r"\W+", "_", Path(filename).stem, flags=re.UNICODE).strip("_") or "data"
    if table[0].isdigit():
        table = f"t_{table}"
    try:
        with sqlite3.connect(path) as connection:
            frame.to_sql(table, connection, index=False)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"dataset_id": dataset_id, "filename": filename, **get_schema(dataset_id)}


def get_schema(dataset_id: str) -> dict:
    with sqlite3.connect(db_path(dataset_id)) as connection:
        tables = []
        for (table,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ):
            columns = [
                {"name": name, "type": (kind or "TEXT").upper()}
                for _, name, kind, *_ in connection.execute(
                    f"PRAGMA table_info({qi(table)})"
                )
            ]
            rows = connection.execute(f"SELECT * FROM {qi(table)} LIMIT 3").fetchall()
            tables.append({
                "name": table,
                "row_count": connection.execute(
                    f"SELECT COUNT(*) FROM {qi(table)}"
                ).fetchone()[0],
                "columns": columns,
                "sample_rows": rows,
            })
    return {"tables": tables}


def schema_prompt(dataset_id: str) -> str:
    lines = []
    for table in get_schema(dataset_id)["tables"]:
        columns = ", ".join(f"{qi(c['name'])} {c['type']}" for c in table["columns"])
        lines.append(f"Table {qi(table['name'])} ({table['row_count']} rows): {columns}")
        lines.append(f"Sample rows: {json.dumps(table['sample_rows'], ensure_ascii=False, default=str)}")
    return "\n".join(lines)

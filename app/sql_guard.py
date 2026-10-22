"""Validate and execute LLM-generated SQLite queries safely."""
import math
import re
import sqlite3
import time

from .db import db_path, get_schema

MAX_ROWS = 500
TIMEOUT = 10
_FORBIDDEN = re.compile(
    r"\b(attach|detach|pragma|insert|update|delete|drop|create|alter|vacuum|"
    r"reindex|truncate|load_extension|savepoint|commit|rollback|begin)\b", re.I)
_QUOTED = re.compile(r'"((?:[^"]|"")*)"')
_SQL_TEXT = re.compile(r"--[^\n]*|/\*.*?\*/|'(?:[^']|'')*'|`[^`]*`", re.S)


class SQLValidationError(ValueError):
    pass


class SQLExecutionError(RuntimeError):
    pass


def validate_sql(sql: str) -> str:
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        raise SQLValidationError("The model returned an empty query")
    clean = _SQL_TEXT.sub(" ", sql)
    if ";" in clean:
        raise SQLValidationError("Only one SQL statement is allowed")
    if not re.match(r"^(select|with)\b", sql, re.I):
        raise SQLValidationError("Only SELECT queries are allowed")
    if _FORBIDDEN.search(clean):
        raise SQLValidationError("The query contains a forbidden command")
    return sql


def _authorizer(action, *_):
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def _check_identifiers(dataset_id: str, sql: str) -> None:
    known = {table["name"].casefold() for table in get_schema(dataset_id)["tables"]}
    for table in get_schema(dataset_id)["tables"]:
        known.update(column["name"].casefold() for column in table["columns"])
    aliases = re.findall(r'\bAS\s+"((?:[^"]|"")*)"', sql, re.I)
    known.update(alias.replace('""', '"').casefold() for alias in aliases)
    for identifier in _QUOTED.findall(sql):
        name = identifier.replace('""', '"').casefold()
        if name not in known:
            raise SQLExecutionError(f'no such column or table: "{identifier}"')


def _json_value(value):
    if isinstance(value, bytes):
        return f"<binary {len(value)} bytes>"
    return None if isinstance(value, float) and not math.isfinite(value) else value


def execute_readonly(dataset_id: str, sql: str, max_rows: int = MAX_ROWS,
                     timeout: float = TIMEOUT) -> dict:
    sql = validate_sql(sql)
    connection = sqlite3.connect(
        db_path(dataset_id).resolve().as_uri() + "?mode=ro", uri=True)
    _check_identifiers(dataset_id, sql)
    connection.set_authorizer(_authorizer)
    deadline = time.monotonic() + timeout
    connection.set_progress_handler(
        lambda: int(time.monotonic() > deadline), 10000)
    try:
        cursor = connection.execute(sql)
        columns = [column[0] for column in cursor.description or ()]
        rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.Error as exc:
        message = "Query timed out" if "interrupted" in str(exc).lower() else str(exc)
        raise SQLExecutionError(message) from exc
    finally:
        connection.close()
    return {
        "columns": columns,
        "rows": [[_json_value(value) for value in row] for row in rows[:max_rows]],
        "row_count": min(len(rows), max_rows),
        "truncated": len(rows) > max_rows,
    }

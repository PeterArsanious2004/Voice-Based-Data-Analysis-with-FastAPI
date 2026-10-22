"""Voice question -> LangChain SQL -> SQLite result."""
import re

from langchain_core.messages import HumanMessage, SystemMessage

from .db import schema_prompt
from .llm import get_llm
from .sql_guard import execute_readonly, validate_sql

SQL_PROMPT = """Convert the user's question into one safe SQLite SELECT query.
The question may be Arabic or English. Use only the tables and columns below.
Quote every table and column name with double quotes. Return SQL only.

Schema:
{schema}"""


def _sql_from_response(response: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", response, re.S | re.I)
    return validate_sql((match.group(1) if match else response).strip())


def answer_question(dataset_id: str, question: str) -> dict:
    """Generate one query with LangChain, execute it, and return its rows."""
    response = get_llm().invoke([
        SystemMessage(content=SQL_PROMPT.format(schema=schema_prompt(dataset_id))),
        HumanMessage(content=question),
    ])
    sql = _sql_from_response(str(response.content))
    return {"question": question, "sql": sql, **execute_readonly(dataset_id, sql)}

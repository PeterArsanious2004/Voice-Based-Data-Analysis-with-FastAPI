"""Five-step voice data analysis API."""
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import db, sql_agent, stt
from .sql_guard import SQLExecutionError, SQLValidationError

STATIC = Path(__file__).parent / "static"
MAX_DATASET_BYTES = 50 * 1024 * 1024
MAX_AUDIO_BYTES = 25 * 1024 * 1024

app = FastAPI(title="Voice Data Analyst",
              description="Voice -> Faster-Whisper -> LangChain -> SQLite -> result")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(db.DatasetNotFound)
@app.exception_handler(SQLValidationError)
@app.exception_handler(SQLExecutionError)
async def handle_error(_, exc):
    status = 404 if isinstance(exc, db.DatasetNotFound) else 400
    return JSONResponse(status_code=status, content={"detail": str(exc)})


def _read_file(upload: UploadFile, limit: int) -> bytes:
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, "File is too large")
    if not data:
        raise HTTPException(400, "File is empty")
    return data


def _transcribe(audio: UploadFile, language: str | None) -> dict:
    data = _read_file(audio, MAX_AUDIO_BYTES)
    suffix = Path(audio.filename or "question.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(data)
    try:
        result = stt.transcribe(temp.name, None if not language or language == "auto" else language)
    except Exception as exc:
        raise HTTPException(400, f"Speech recognition failed: {exc}") from exc
    finally:
        os.unlink(temp.name)
    if not result["text"]:
        raise HTTPException(422, "No speech detected")
    return result


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/datasets", summary="Upload the dataset")
def upload_dataset(file: UploadFile = File(...)):
    try:
        return db.create_dataset(file.filename or "data.csv", _read_file(file, MAX_DATASET_BYTES))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/analyze", summary="Voice -> text -> SQL -> SQLite -> result")
def analyze(dataset_id: str = Form(...), audio: UploadFile = File(...),
            language: str | None = Form(None)):
    db.db_path(dataset_id)
    transcript = _transcribe(audio, language)
    result = sql_agent.answer_question(dataset_id, transcript["text"])
    return {"transcript": transcript, **result}

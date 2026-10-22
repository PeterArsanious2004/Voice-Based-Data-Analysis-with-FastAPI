"""Speech-to-text used by the FastAPI voice endpoint."""
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu")


def transcribe(path: str, language: str | None = None) -> dict:
    segments, info = model.transcribe(
        path,
        language=language,
        task="transcribe",
        beam_size=1,
        vad_filter=True,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return {
        "text": text,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "duration": round(float(info.duration), 2),
    }

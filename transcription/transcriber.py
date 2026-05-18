import os
from faster_whisper import WhisperModel

_model: WhisperModel | None = None


def _load_model() -> WhisperModel:
    global _model
    if _model is None:
        model_size = os.getenv("WHISPER_MODEL", "small")
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


def transcribe_file(file_path: str) -> str:
    """Transcribe an audio file and return the full text."""
    model = _load_model()
    segments, _ = model.transcribe(
        file_path,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    return " ".join(s.text for s in segments).strip()

import os
import queue
import tempfile
import threading
import wave
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
BLOCK_SECONDS = 3
CHANNELS = 1

_model: WhisperModel | None = None


def _load_model() -> WhisperModel:
    global _model
    if _model is None:
        model_size = os.getenv("WHISPER_MODEL", "small")
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


class LiveTranscriber:
    """
    Streams microphone audio, transcribes in real time, and saves the full
    session to a WAV file for post-hoc speaker diarization.
    """

    def __init__(self, on_text):
        self.on_text = on_text
        self._audio_q: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: sd.InputStream | None = None

        # accumulate all raw audio for diarization
        self._all_audio: list[np.ndarray] = []
        self._audio_lock = threading.Lock()

        # path to saved WAV after stop()
        self.wav_path: str | None = None

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata.copy()
        self._audio_q.put(chunk)
        with self._audio_lock:
            self._all_audio.append(chunk.flatten())

    def _process_loop(self):
        model = _load_model()
        buffer = np.empty((0,), dtype=np.float32)
        chunk_size = SAMPLE_RATE * BLOCK_SECONDS

        while not self._stop_event.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.5)
                buffer = np.concatenate([buffer, chunk.flatten()])
            except queue.Empty:
                continue

            if len(buffer) >= chunk_size:
                audio_chunk = buffer[:chunk_size].astype(np.float32)
                buffer = buffer[chunk_size:]
                segments, _ = model.transcribe(
                    audio_chunk,
                    language="en",
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 300},
                )
                text = " ".join(s.text for s in segments).strip()
                if text:
                    self.on_text(text)

        # flush remaining audio
        if len(buffer) > SAMPLE_RATE:
            segments, _ = _load_model().transcribe(
                buffer.astype(np.float32), language="en", vad_filter=True
            )
            text = " ".join(s.text for s in segments).strip()
            if text:
                self.on_text(text)

    def start(self):
        self._stop_event.clear()
        self._all_audio.clear()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=SAMPLE_RATE,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self) -> str | None:
        """Stop recording and save full audio to a WAV file. Returns the WAV path."""
        self._stop_event.set()
        if self._stream:
            self._stream.stop()
            self._stream.close()
        if self._thread:
            self._thread.join(timeout=5)

        with self._audio_lock:
            all_audio = list(self._all_audio)

        if not all_audio:
            return None

        full_audio = np.concatenate(all_audio).astype(np.float32)
        pcm = (full_audio * 32767).astype(np.int16)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

        self.wav_path = tmp.name
        return tmp.name

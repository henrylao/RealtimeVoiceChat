"""Opt-in per-turn audio collection for development (env: AUDIO_DUMP=true).

Captures, per conversation turn:
  - user_turn_NN.wav      16 kHz mono, exactly what STT consumed that turn
  - assistant_turn_NN.wav 24 kHz mono, TTS PCM as streamed to the client
  - manifest.jsonl        one JSON object per event linking audio, transcripts,
                          latencies, speaker scores and the session config

All writes happen on background threads; the audio path never blocks on disk.
Disabled by default (no-op when the flag is off). Retention pruning keeps the
dump directory under configured size/age caps. Local disk only.
"""

import json
import logging
import os
import threading
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

USER_SR = 16000
TTS_SR = 24000


def _write_wav(path: Path, pcm16: np.ndarray, sample_rate: int) -> None:
    pcm = np.ascontiguousarray(pcm16, dtype=np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


class AudioDumper:
    """Per-turn audio + manifest collection. Thread-safe, background writes."""

    def __init__(
        self,
        dump_dir: Optional[str] = None,
        max_gb: float = float(os.getenv("AUDIO_DUMP_RETENTION_GB", "50")),
        max_days: float = float(os.getenv("AUDIO_DUMP_RETENTION_DAYS", "30")),
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        root = Path(dump_dir) if dump_dir else BASE_DIR.parent / "data" / "sessions"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_dir = root / stamp
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb * 1024 ** 3)
        self.max_days = max_days
        self._turn = 0
        self._lock = threading.Lock()
        manifest_path = self.session_dir / "manifest.jsonl"
        self._manifest = open(manifest_path, "a", encoding="utf-8")
        self.log_event({"event": "session_start", "config": config or {}})
        # Prune old sessions in the background so boot is never blocked.
        threading.Thread(target=self._prune, name="AudioDumpPrune", daemon=True).start()
        logger.info(f"💾 Audio dump session: {self.session_dir}")

    # -- public API ----------------------------------------------------
    def next_turn(self) -> int:
        with self._lock:
            self._turn += 1
            return self._turn

    @property
    def current_turn(self) -> int:
        with self._lock:
            return self._turn

    def save_user_turn(self, turn: int, pcm16: np.ndarray) -> Optional[str]:
        """Persist 16 kHz mono user-turn audio. Returns the file name."""
        if pcm16 is None or len(pcm16) == 0:
            return None
        name = f"user_turn_{turn:02d}.wav"
        self._write_async(self.session_dir / name, np.asarray(pcm16, dtype=np.int16), USER_SR)
        return name

    def save_assistant_turn(self, turn: int, chunks: List[bytes]) -> Optional[str]:
        """Persist 24 kHz mono assistant-turn audio from TTS chunks."""
        if not chunks:
            return None
        try:
            pcm = np.frombuffer(b"".join(chunks), dtype=np.int16)
        except Exception as e:
            logger.warning(f"💾 assistant turn assembly failed: {e}")
            return None
        if len(pcm) == 0:
            return None
        name = f"assistant_turn_{turn:02d}.wav"
        self._write_async(self.session_dir / name, pcm, TTS_SR)
        return name

    def log_event(self, event: Dict[str, Any]) -> None:
        try:
            with self._lock:
                record = {"ts": datetime.now(timezone.utc).isoformat(), **event}
                self._manifest.write(json.dumps(record) + "\n")
                self._manifest.flush()
        except Exception as e:
            logger.warning(f"💾 manifest write failed: {e}")

    # -- internals -----------------------------------------------------
    def _write_async(self, path: Path, pcm16: np.ndarray, sample_rate: int) -> None:
        def _do() -> None:
            try:
                _write_wav(path, pcm16, sample_rate)
                logger.info(f"💾 saved {path.name} ({len(pcm16)/sample_rate:.1f}s)")
            except Exception as e:
                logger.warning(f"💾 wav write failed for {path.name}: {e}")

        threading.Thread(target=_do, name=f"AudioDump-{path.name}", daemon=True).start()

    def _prune(self) -> None:
        """Delete oldest sessions until under size/age caps. Best effort."""
        try:
            root = self.session_dir.parent
            if not root.exists():
                return
            now = datetime.now(timezone.utc).timestamp()
            sessions = sorted(
                [d for d in root.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
            )
            for d in sessions:
                age_days = (now - d.stat().st_mtime) / 86400.0
                if d != self.session_dir and age_days > self.max_days:
                    self._rm_session(d, f"older than {self.max_days}d")
            total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            for d in sessions:
                if total <= self.max_bytes:
                    break
                if d == self.session_dir:
                    continue
                total -= sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
                self._rm_session(d, "over size cap")
        except Exception as e:
            logger.warning(f"💾 retention prune skipped: {e}")

    @staticmethod
    def _rm_session(d: Path, reason: str) -> None:
        try:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
            logger.info(f"💾 pruned old session {d.name} ({reason})")
        except Exception:
            pass


_dumper: Optional[AudioDumper] = None
_dumper_lock = threading.Lock()


def get_dumper(config: Optional[Dict[str, Any]] = None) -> Optional[AudioDumper]:
    """Return the shared dumper, or None when AUDIO_DUMP is not enabled."""
    if os.getenv("AUDIO_DUMP", "false").lower() not in ("1", "true", "yes"):
        return None
    global _dumper
    with _dumper_lock:
        if _dumper is None:
            dump_dir = os.getenv("AUDIO_DUMP_DIR") or None
            _dumper = AudioDumper(dump_dir=dump_dir, config=config)
        return _dumper

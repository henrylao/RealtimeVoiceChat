"""Rolling speaker verification for the voice pipeline (flag-only observability).

Uses TitaNet-Large via the ``speakeronnx`` library (pure onnxruntime, CPU) to
score incoming 16 kHz audio windows against an enrolled owner profile.

Design: pinned anchor profile (never modified) + rolling EMA profile that
adapts only on high-confidence accepts, with a drift guard that resets the
rolling profile back to the anchor if it wanders too far. Verification runs
off the STT critical path; callers only read ``speaker_score`` metadata.

No NeMo dependency. See plan notes for threshold calibration procedure.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from speakeronnx import SpeakerEmbedder, cosine as _cosine
    SPEAKERONNX_AVAILABLE = True
except ImportError:
    SPEAKERONNX_AVAILABLE = False
    SpeakerEmbedder = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = os.getenv("SPEAKER_MODEL", "titanet-large")
# Measured 2026-09-28 on dump data (mic anchor): owner 0.70-0.83, impostor
# 0.20-0.23. Thin eval (4 clips, one sitting) — recalibrate with more data
# before enforcing.
DEFAULT_VERIFY_THRESHOLD = float(os.getenv("SPEAKER_VERIFY_THRESHOLD", "0.45"))
DEFAULT_ADAPT_THRESHOLD = float(os.getenv("SPEAKER_ADAPT_THRESHOLD", "0.8"))
DEFAULT_ADAPT_ALPHA = float(os.getenv("SPEAKER_ADAPT_ALPHA", "0.05"))
DEFAULT_MIN_ANCHOR_SIM = float(os.getenv("SPEAKER_MIN_ANCHOR_SIM", "0.5"))
DEFAULT_MIN_WINDOW_S = float(os.getenv("SPEAKER_MIN_WINDOW_S", "1.0"))
SAMPLE_RATE = 16000


class SpeakerVerifier:
    """Rolling speaker verification against an enrolled owner profile."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        profile_path: Optional[str] = None,
        verify_threshold: float = DEFAULT_VERIFY_THRESHOLD,
        adapt_threshold: float = DEFAULT_ADAPT_THRESHOLD,
        adapt_alpha: float = DEFAULT_ADAPT_ALPHA,
        min_anchor_sim: float = DEFAULT_MIN_ANCHOR_SIM,
        min_window_s: float = DEFAULT_MIN_WINDOW_S,
    ) -> None:
        if not SPEAKERONNX_AVAILABLE:
            raise ImportError("speakeronnx is required for SpeakerVerifier but not installed.")
        self.model_name = model
        self.verify_threshold = verify_threshold
        self.adapt_threshold = adapt_threshold
        self.adapt_alpha = adapt_alpha
        self.min_anchor_sim = min_anchor_sim
        self.min_window_s = min_window_s
        self.profile_path = Path(profile_path) if profile_path else BASE_DIR / "owner_profile.npy"

        self._lock = threading.Lock()
        logger.info(f"🔐 Loading speaker model '{model}' (onnxruntime CPU)...")
        self.embedder = SpeakerEmbedder(model=model)
        self.anchor: Optional[np.ndarray] = None
        self.rolling: Optional[np.ndarray] = None
        self._load_or_enroll_default()

    def _load_or_enroll_default(self) -> None:
        if self.profile_path.exists():
            self.anchor = np.load(str(self.profile_path)).astype(np.float32)
            self.rolling = self.anchor.copy()
            logger.info(f"🔐 Loaded owner profile from {self.profile_path}")
        else:
            logger.warning(f"🔐 No owner profile at {self.profile_path}; call enroll_file() to create one.")

    def enroll_file(self, wav_path: str) -> np.ndarray:
        """Enroll the owner from a wav file; persists the pinned anchor profile."""
        emb = self.embedder.embed(str(wav_path)).astype(np.float32)
        with self._lock:
            self.anchor = emb.copy()
            self.rolling = emb.copy()
            np.save(str(self.profile_path), self.anchor)
        logger.info(f"🔐 Enrolled owner profile from {wav_path} -> {self.profile_path}")
        return emb

    def enroll_pcm(self, pcm16: np.ndarray) -> np.ndarray:
        """Enroll from int16 PCM samples at 16 kHz (for live enrollment later)."""
        audio = pcm16.astype(np.float32) / 32768.0
        emb = self.embedder.embed(audio).astype(np.float32)
        with self._lock:
            self.anchor = emb.copy()
            self.rolling = emb.copy()
            np.save(str(self.profile_path), self.anchor)
        logger.info("🔐 Enrolled owner profile from live PCM")
        return emb

    def verify_window(self, pcm16: np.ndarray) -> dict:
        """Score a 16 kHz int16 PCM window. Returns score metadata (flag-only).

        Windows shorter than ``min_window_s`` return ``reliable: False``.
        """
        dur_s = len(pcm16) / SAMPLE_RATE
        if self.anchor is None:
            return {"speaker_score": 0.0, "is_owner": False, "reliable": False,
                    "reason": "no-profile", "window_s": round(dur_s, 2)}
        if dur_s < self.min_window_s:
            return {"speaker_score": 0.0, "is_owner": False, "reliable": False,
                    "reason": "too-short", "window_s": round(dur_s, 2)}
        try:
            audio = pcm16.astype(np.float32) / 32768.0
            emb = self.embedder.embed(audio).astype(np.float32)
        except Exception as e:
            logger.warning(f"🔐 embedding failed: {e}")
            return {"speaker_score": 0.0, "is_owner": False, "reliable": False,
                    "reason": "embed-error", "window_s": round(dur_s, 2)}

        with self._lock:
            anchor = self.anchor
            rolling = self.rolling
        score_anchor = float(_cosine(emb, anchor))
        score_rolling = float(_cosine(emb, rolling)) if rolling is not None else score_anchor
        score = max(score_anchor, score_rolling)
        is_owner = score >= self.verify_threshold

        # Rolling adaptation: only on confident accepts, drift-guarded.
        if is_owner and score >= self.adapt_threshold and rolling is not None:
            with self._lock:
                new_rolling = (1.0 - self.adapt_alpha) * self.rolling + self.adapt_alpha * emb
                new_rolling = new_rolling / (np.linalg.norm(new_rolling) + 1e-9)
                if float(_cosine(new_rolling, self.anchor)) < self.min_anchor_sim:
                    self.rolling = self.anchor.copy()
                    logger.warning("🔐 Rolling profile drifted; reset to anchor.")
                else:
                    self.rolling = new_rolling.astype(np.float32)

        return {"speaker_score": round(score, 4),
                "score_anchor": round(score_anchor, 4),
                "score_rolling": round(score_rolling, 4),
                "is_owner": is_owner,
                "reliable": True,
                "reason": "ok",
                "window_s": round(dur_s, 2)}

    def verify_segments(self, pcm16: np.ndarray, window_s: float = 5.0,
                        hop_s: float = 2.5) -> list:
        """Score overlapping fixed windows across a (possibly long) turn.

        Long turns are sequences, not single speaker events: a full-turn
        average hides mixed content (e.g. background media with the owner
        joining at the tail). Returns per-segment verify_window dicts with
        start/end offsets. Segments shorter than min_window_s are skipped.
        """
        sr = SAMPLE_RATE
        win, hop = int(window_s * sr), int(hop_s * sr)
        min_len = int(self.min_window_s * sr)
        if len(pcm16) < min_len:
            return []
        starts = list(range(0, max(1, len(pcm16) - win + 1), hop))
        covered_end = starts[-1] + win if starts else 0
        if len(pcm16) - covered_end >= min_len and len(pcm16) - win != starts[-1]:
            starts.append(len(pcm16) - win)  # end-anchored tail window
        out = []
        for start in starts:
            seg = pcm16[start:start + win]
            if len(seg) < min_len:
                continue
            r = self.verify_window(seg)
            r["start_s"] = round(start / sr, 2)
            r["end_s"] = round(min(start + len(seg), len(pcm16)) / sr, 2)
            out.append(r)
        return out

    def verify_turn(self, pcm16: np.ndarray, window_s: float = 5.0,
                    hop_s: float = 2.5, margin_band: float = 0.1,
                    registry=None) -> dict:
        """Roll segment verdicts into one turn decision.

        With a registry (recommended), each segment is attributed via group
        best-match + margin: unanimous confident G1 -> "obey", unanimous
        confident G2/unknown -> "ack", anything split or thin -> "uncertain".
        Without a registry, falls back to anchor-threshold verdicts (weaker:
        same-room media can clear an absolute threshold; only the margin
        against alternatives separates it).
        """
        if self.anchor is None:
            return {"decision": "uncertain", "reason": "no-profile", "segments": []}
        segments = self.verify_segments(pcm16, window_s, hop_s)
        reliable = [s for s in segments if s.get("reliable")]
        if not reliable:
            return {"decision": "uncertain", "reason": "no-reliable-segments",
                    "segments": segments}
        if registry is not None:
            return self._rollup_group(pcm16, reliable, registry, margin_band)
        owner_flags = [bool(s["is_owner"]) for s in reliable]
        margins = [abs(float(s["speaker_score"]) - self.verify_threshold)
                   for s in reliable]
        confident = all(m >= margin_band for m in margins)
        if all(owner_flags) and confident:
            decision, reason = "obey", "unanimous-owner-confident"
        elif not any(owner_flags) and confident:
            decision, reason = "ack", "unanimous-non-owner-confident"
        else:
            decision, reason = "uncertain", "split-or-thin-margin"
        return {"decision": decision, "reason": reason,
                "owner_fraction": round(sum(owner_flags) / len(owner_flags), 3),
                "segments": segments}

    def _rollup_group(self, pcm16: np.ndarray, reliable: list, registry,
                      margin_band: float) -> dict:
        """Attribute each segment to a registry group; roll up with margins.

        Segment embeddings are recomputed from stored offsets (cheap relative
        to getting the attribution level wrong by reusing anchor scores).
        """
        import numpy as np
        votes = []
        for s in reliable:
            start = int(s["start_s"] * SAMPLE_RATE)
            end = int(s["end_s"] * SAMPLE_RATE)
            seg = pcm16[start:end].astype(np.float32) / 32768.0
            try:
                seg_emb = self.embedder.embed(seg).astype(np.float32)
            except Exception:
                votes.append(("uncertain", 0.0))
                continue
            vid, grp, best, margin = registry.identify(seg_emb)
            if grp == "G1" and margin >= margin_band:
                votes.append(("obey", margin))
            elif margin >= margin_band:
                votes.append(("ack", margin))
            else:
                votes.append(("uncertain", margin))
        kinds = [v for v, _ in votes]
        n_obey = sum(1 for k in kinds if k == "obey")
        n_ack = sum(1 for k in kinds if k == "ack")
        # Confident segments vote; uncertain ones abstain. Any confident
        # opposition forces "uncertain" (split turn); unopposed confident
        # votes decide; all-abstain stays uncertain.
        if n_obey > 0 and n_ack == 0:
            decision, reason = "obey", "unopposed-owner-evidence"
        elif n_ack > 0 and n_obey == 0:
            decision, reason = "ack", "unopposed-non-owner-evidence"
        else:
            decision, reason = "uncertain", "conflicting-or-no-confident-evidence"
        return {"decision": decision, "reason": reason,
                "segment_votes": kinds,
                "segments": [{k: s[k] for k in s if not k.startswith("_")}
                             for s in reliable]}


_verifier: Optional[SpeakerVerifier] = None
_verifier_lock = threading.Lock()


def get_verifier() -> Optional[SpeakerVerifier]:
    """Lazily create the shared verifier; None if unavailable/disabled."""
    global _verifier
    if os.getenv("SPEAKER_VERIFY", "true").lower() in ("0", "false", "no"):
        return None
    with _verifier_lock:
        if _verifier is None:
            try:
                _verifier = SpeakerVerifier()
            except Exception as e:
                logger.warning(f"🔐 SpeakerVerifier unavailable: {e}")
                return None
        return _verifier

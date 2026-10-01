"""Versioned voice registry for group-based authorization.

Each entry maps a voiceprint centroid to an authorization group:
  - G1: obey (owner-confident; full answers + actions when enforced)
  - G2: ack-only (heard, never obeyed)

Unknown voices (best match below threshold) fail closed to G2 behavior.
G1 membership is admin-enrolled only; the unknown gallery proposes, the
admin disposes. Registry lives in speaker_id/registry/ as registry.json
plus per-entry embedding .npy files — append-only history, never blind
overwrite (each save bumps version and retains prior files).
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REGISTRY_DIR = Path(os.getenv("VOICE_REGISTRY_DIR", HERE / "registry"))
ENTRIES_FILE = REGISTRY_DIR / "registry.json"
EMB_DIR = REGISTRY_DIR / "embeddings"

OBEY_GROUP = "G1"
ACK_GROUP = "G2"
UNKNOWN = "unknown"
DEFAULT_MATCH_THRESHOLD = float(os.getenv("VOICE_MATCH_THRESHOLD", "0.45"))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class VoiceRegistry:
    """Load/save/query the versioned voice registry."""

    def __init__(self, registry_dir: Optional[Path] = None) -> None:
        self.dir = Path(registry_dir) if registry_dir else REGISTRY_DIR
        self.emb_dir = self.dir / "embeddings"
        self.file = self.dir / "registry.json"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.emb_dir.mkdir(parents=True, exist_ok=True)
        self._entries: List[dict] = []
        self._vectors: Dict[str, np.ndarray] = {}
        self.reload()

    # -- persistence -------------------------------------------------
    def reload(self) -> None:
        self._entries = []
        self._vectors = {}
        if not self.file.exists():
            return
        data = json.loads(self.file.read_text(encoding="utf-8"))
        for e in data.get("entries", []):
            vec_path = self.emb_dir / e["embedding_file"]
            if not vec_path.exists():
                logger.warning(f"🔐 registry entry '{e['id']}' missing {vec_path.name}; skipped")
                continue
            self._entries.append(e)
            self._vectors[e["id"]] = np.load(str(vec_path)).astype(np.float32)

    def _save(self) -> None:
        payload = {"version": _utcnow(), "entries": self._entries}
        self.file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_entry(self, voice_id: str, name: str, group: str, embedding: np.ndarray,
                  clips: List[str], enrolled_by: str = "admin") -> dict:
        """Append (or replace by id, versioned) a registry entry."""
        if group not in (OBEY_GROUP, ACK_GROUP):
            raise ValueError(f"group must be {OBEY_GROUP} or {ACK_GROUP}")
        emb = np.asarray(embedding, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        fname = f"{voice_id}_{stamp}.npy"
        np.save(str(self.emb_dir / fname), emb)
        entry = {"id": voice_id, "name": name, "group": group,
                 "embedding_file": fname, "clips": clips,
                 "enrolled_by": enrolled_by, "enrolled_at": _utcnow()}
        self._entries = [e for e in self._entries if e["id"] != voice_id] + [entry]
        self._vectors[voice_id] = emb
        self._save()
        logger.info(f"🔐 registry: {voice_id} ({name}) -> {group}")
        return entry

    # -- query -------------------------------------------------------
    def identify(self, embedding: np.ndarray,
                 threshold: float = DEFAULT_MATCH_THRESHOLD
                 ) -> Tuple[str, Optional[str], float, float]:
        """Best-match voice id + group, or (unknown, None) below threshold.

        Returns (voice_id, group, best_score, margin_over_runner_up).
        Unknowns fail closed: group None means treat as G2.
        """
        from speakeronnx import cosine
        emb = np.asarray(embedding, dtype=np.float32)
        scored = sorted(
            ((vid, float(cosine(emb, vec))) for vid, vec in self._vectors.items()),
            key=lambda t: -t[1],
        )
        if not scored or scored[0][1] < threshold:
            best = scored[0][1] if scored else 0.0
            return UNKNOWN, None, best, 0.0
        (vid, best) = scored[0]
        runner = scored[1][1] if len(scored) > 1 else 0.0
        group = next(e["group"] for e in self._entries if e["id"] == vid)
        return vid, group, best, best - runner

    def summary(self) -> List[dict]:
        return [{"id": e["id"], "name": e["name"], "group": e["group"],
                 "clips": len(e["clips"]), "enrolled_at": e["enrolled_at"]}
                for e in self._entries]

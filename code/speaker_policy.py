"""Per-turn speaker policy: hear everyone, obey only the owner.

Decision levels (flag-only unless SPEAKER_ENFORCE=true):
  - "obey"   : owner-confident turn; full answer + actions allowed.
  - "ack"    : heard but not owner-confident; acknowledge briefly, never act.
               ("Ack but never obey" — silence is itself a response.)
  - "silent" : reserved for future use (currently never returned).

Unreliable scores (too-short window, no profile, embed error) fail open to
"obey": without evidence, the turn is treated as the owner. Uncertainty that
should restrict behavior is expressed by callers, not here.

Thresholds via env: SPEAKER_OBEY_THRESHOLD (default 0.45, measured).
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

OBEY_THRESHOLD = float(os.getenv("SPEAKER_OBEY_THRESHOLD", "0.45"))
ENFORCE = os.getenv("SPEAKER_ENFORCE", "false").lower() in ("1", "true", "yes")


def decide(speaker_score: Optional[dict]) -> Dict[str, Any]:
    """Decide the policy action for a turn from its speaker score metadata."""
    if not speaker_score or not speaker_score.get("reliable"):
        return {"action": "obey", "reason": "no-score-fail-open",
                "score": None}
    score = float(speaker_score.get("speaker_score", 0.0))
    if score >= OBEY_THRESHOLD:
        return {"action": "obey", "reason": "owner-confident",
                "score": score}
    return {"action": "ack", "reason": "not-owner-confident",
            "score": score}


def should_enforce() -> bool:
    """True only when policy decisions drive behavior (default: log only)."""
    return ENFORCE

"""Admin enrollment of a voice into an authorization group.

Usage:
    ../.venv/bin/python enroll_voice.py --name owner --group G1 --clip <wav> [--voice-id owner]
    ../.venv/bin/python enroll_voice.py --name kokoro-tts --group G2 --clip <wav>

Validates clip quality (exists, 2-30s, embeds cleanly), appends a versioned
registry entry, then re-runs eval.py so margin regressions are visible before
the entry goes live. G1 membership is admin-only by design.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from registry import VoiceRegistry, OBEY_GROUP, ACK_GROUP  # noqa: E402
from speaker_verify import SpeakerVerifier, SAMPLE_RATE  # noqa: E402

MIN_CLIP_S = 2.0
WARN_CLIP_S = 30.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Enroll a voice into G1/G2.")
    ap.add_argument("--name", required=True, help="Human label, e.g. owner, kokoro-tts")
    ap.add_argument("--group", required=True, choices=[OBEY_GROUP, ACK_GROUP])
    ap.add_argument("--clip", required=True, help="WAV file (any rate; resampled to 16k)")
    ap.add_argument("--voice-id", default=None, help="Registry id (default: name slug)")
    args = ap.parse_args()

    clip = Path(args.clip)
    if not clip.exists():
        print(f"ERROR: clip not found: {clip}")
        return 1
    info = sf.info(str(clip))
    dur = info.frames / info.samplerate
    print(f"clip: {dur:.1f}s, {info.samplerate}Hz, {info.channels}ch")
    if dur < MIN_CLIP_S:
        print(f"ERROR: clip shorter than {MIN_CLIP_S}s minimum — too little voice info.")
        return 1
    if dur > WARN_CLIP_S:
        print(f"NOTE: clip longer than {WARN_CLIP_S}s; using first {WARN_CLIP_S}s.")
    if info.channels != 1:
        print("NOTE: stereo will be downmixed to mono.")

    voice_id = args.voice_id or "".join(c if c.isalnum() else "-" for c in args.name.lower())
    v = SpeakerVerifier()
    seg = load16k_head(str(clip), WARN_CLIP_S)
    emb = v.embedder.embed((seg.astype(np.float32) / 32768.0)).astype(np.float32)
    if float(np.linalg.norm(emb)) < 1e-6:
        print("ERROR: empty embedding — clip has no usable speech?")
        return 1

    reg = VoiceRegistry()
    reg.add_entry(voice_id, args.name, args.group, emb, clips=[str(clip)])
    print(f"enrolled '{voice_id}' ({args.name}) -> {args.group}")
    print("\nregistry:")
    for s in reg.summary():
        print(f"  {s['id']} ({s['name']}) -> {s['group']} [{s['clips']} clips]")

    print("\nre-running eval to check for margin regression...")
    r = subprocess.run([sys.executable, str(HERE / "eval.py")],
                       capture_output=True, text=True, timeout=600)
    tail = [l for l in r.stdout.splitlines() if l.strip()][-12:]
    print("\n".join(tail))
    return 0


def load16k_head(path: str, max_s: float) -> np.ndarray:
    import scipy.signal
    pcm, sr = sf.read(path, dtype="int16")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    if sr != SAMPLE_RATE:
        n = int(len(pcm) * SAMPLE_RATE / sr)
        pcm = scipy.signal.resample(pcm.astype(np.float32), n).astype(np.int16)
    return pcm[: int(max_s * SAMPLE_RATE)]


if __name__ == "__main__":
    raise SystemExit(main())

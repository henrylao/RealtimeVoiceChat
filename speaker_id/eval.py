"""Standalone evaluation for the speaker verification prototype.

Uses collected dump data (data/sessions/*/user_turn_*.wav as owner clips,
assistant_turn_*.wav as impostor clips) to test:
  1. enrollment consistency (mic clip profile vs MP3 reference profile)
  2. owner acceptance across all user turns
  3. impostor rejection on all assistant (TTS) turns
  4. window-length sensitivity (1s / 2s / 3s slices)
  5. measured threshold proposal from score distributions
  6. rolling EMA profile behavior across sequential turns

Usage: cd speaker_id && ../.venv/bin/python eval.py
Read-only w.r.t. the voice server; writes nothing except stdout.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from speaker_verify import SpeakerVerifier, SAMPLE_RATE  # noqa: E402
from speakeronnx import cosine  # noqa: E402


def load16k(path: Path) -> np.ndarray:
    pcm, sr = sf.read(str(path), dtype="int16")
    if sr != SAMPLE_RATE:
        import scipy.signal
        n = int(len(pcm) * SAMPLE_RATE / sr)
        pcm = scipy.signal.resample(pcm.astype(np.float32), n).astype(np.int16)
    return pcm


def find_dumps():
    users, assistants = [], []
    # Known false-positive turns (background media, not the owner) are excluded
    # from the owner set; they are covered in section [7] instead.
    exclude_owners = {
        "20260928T052656Z/user_turn_02.wav",
        "20260928T052656Z/user_turn_03.wav",
        "20260928T052656Z/user_turn_04.wav",
        "20260928T052656Z/user_turn_05.wav",  # ambiguous ("One thing that I really...")
    }
    for sess in sorted((REPO / "data" / "sessions").glob("*")):
        for u in sorted(sess.glob("user_turn_*.wav")):
            if f"{sess.name}/{u.name}" not in exclude_owners:
                users.append(u)
        assistants += sorted(sess.glob("assistant_turn_*.wav"))
    return users, assistants


def main() -> int:
    users, assistants = find_dumps()
    print(f"owner clips: {len(users)}, impostor clips: {len(assistants)}")
    if not users:
        print("no dump data found")
        return 1

    v = SpeakerVerifier()
    owner_pcm = load16k(users[0])
    mic_profile = v.embedder.embed((owner_pcm.astype(np.float32) / 32768.0)).astype(np.float32)

    # 1. enrollment consistency: mic profile vs old MP3-reference profile
    mp3ref_path = HERE / "owner_profile.mp3ref.bak.npy"
    if mp3ref_path.exists():
        mp3ref = np.load(str(mp3ref_path)).astype(np.float32)
        ref_sim = float(cosine(mic_profile, mp3ref))
        print(f"\n[1] mic-profile vs MP3-reference profile cosine: {ref_sim:.4f}")
    else:
        print("\n[1] no MP3-reference backup found; skipping")

    # 2. owner acceptance (full clips, 3s head window, 3s tail window)
    print("\n[2] owner acceptance (anchor=enroll_owner):")
    accept_scores = []
    for u in users:
        pcm = load16k(u)
        full = v.verify_window(pcm)
        head = v.verify_window(pcm[: 3 * SAMPLE_RATE])
        accept_scores.append(full["speaker_score"])
        print(f"    {u.parent.name}/{u.name}: full={full['speaker_score']} "
              f"head3s={head['speaker_score']} reliable={full['reliable']}")

    # 3. impostor rejection (assistant TTS turns)
    print("\n[3] impostor rejection (Kokoro TTS vs owner anchor):")
    reject_scores = []
    for a in assistants:
        pcm = load16k(a)
        seg = pcm[: 3 * SAMPLE_RATE] if len(pcm) >= 3 * SAMPLE_RATE else pcm
        r = v.verify_window(seg)
        reject_scores.append(r["speaker_score"])
        print(f"    {a.parent.name}/{a.name}: score={r['speaker_score']} reliable={r['reliable']}")

    # 4. window sensitivity on the longest owner clip
    longest = max(users, key=lambda p: p.stat().st_size)
    pcm = load16k(longest)
    print(f"\n[4] window sensitivity ({longest.parent.name}/{longest.name}):")
    for s in (1, 2, 3, 5):
        seg = pcm[: s * SAMPLE_RATE]
        r = v.verify_window(seg)
        print(f"    {s}s: score={r['speaker_score']} reliable={r['reliable']} reason={r['reason']}")

    # 5. threshold proposal
    print("\n[5] threshold proposal:")
    if accept_scores and reject_scores:
        lo_accept, hi_reject = min(accept_scores), max(reject_scores)
        margin = lo_accept - hi_reject
        proposed = round((lo_accept + hi_reject) / 2, 3)
        print(f"    min(owner)={lo_accept:.4f} max(impostor)={hi_reject:.4f} "
              f"margin={margin:+.4f} proposed_verify={proposed}")
        print("    PASS" if margin > 0 else "    FAIL (overlap — needs more data / better enrollment)")

    # 6. rolling EMA across sequential owner turns
    print("\n[6] rolling profile across sequential owner turns:")
    v2 = SpeakerVerifier()
    for u in users:
        pcm = load16k(u)
        seg = pcm[: 3 * SAMPLE_RATE] if len(pcm) >= 3 * SAMPLE_RATE else pcm
        r = v2.verify_window(seg)
        drift = float(cosine(v2.rolling, v2.anchor))
        print(f"    {u.parent.name}/{u.name}: score={r['speaker_score']} "
              f"rolling-vs-anchor={drift:.4f}")

    # 7. incident session: same-room media bleed vs owner (known hard case).
    # Turn 1 of 20260928T052656Z is the owner; turns 2-4 are background media
    # played in the same room (same mic/channel -> inflated scores).
    # Turn 3 ("One fact.", 3.6s) is ambiguous — short enough to be the owner.
    print("\n[7] incident session 20260928T052656Z (owner vs same-room media):")
    inc = REPO / "data" / "sessions" / "20260928T052656Z"
    if inc.exists():
        for name in ["user_turn_01.wav", "user_turn_02.wav",
                     "user_turn_03.wav", "user_turn_04.wav"]:
            p = inc / name
            if not p.exists():
                continue
            pcm = load16k(p)
            full = v.verify_window(pcm)
            print(f"    {name}: full={full['speaker_score']} "
                  f"dur={len(pcm)/SAMPLE_RATE:.1f}s")
    # 8. group confusion matrix via the voice registry.
    # NOTE: same-room media scores ~0.6 absolute (channel entanglement), which
    # is why group attribution (best-match + margin) replaces single-threshold
    # gating. Labels below encode that design.
    # Labels: user turns in owner set -> G1; assistant turns -> G2;
    # incident media turns (052656Z 02/04 + ambiguous 03/05/06/07) -> G2.
    print("\n[8] group confusion (registry identify, fail-closed unknowns=G2):")
    try:
        from registry import VoiceRegistry
        reg = VoiceRegistry()
        print("    registry:", [(s["id"], s["group"]) for s in reg.summary()])
        media_set = {
            "20260928T052656Z/user_turn_02.wav",
            "20260928T052656Z/user_turn_03.wav",
            "20260928T052656Z/user_turn_04.wav",
            "20260928T052656Z/user_turn_05.wav",
            "20260928T052656Z/user_turn_06.wav",
            "20260928T052656Z/user_turn_07.wav",
        }
        stats = {"G1": {"G1": 0, "G2": 0}, "G2": {"G1": 0, "G2": 0}}
        for sess in sorted((REPO / "data" / "sessions").glob("*")):
            for wav in sorted(sess.glob("user_turn_*.wav")) + sorted(sess.glob("assistant_turn_*.wav")):
                key = f"{sess.name}/{wav.name}"
                if wav.name.startswith("assistant_") or key in media_set:
                    truth = "G2"
                else:
                    truth = "G1"
                pcm = load16k(wav)
                seg = pcm[: 3 * SAMPLE_RATE] if len(pcm) >= 3 * SAMPLE_RATE else pcm
                emb = v.embedder.embed((seg.astype(np.float32) / 32768.0)).astype(np.float32)
                vid, grp, best, margin = reg.identify(emb)
                pred = grp if grp else "G2"  # unknowns fail closed
                stats[truth][pred] += 1
                flag = "" if pred == truth else "  <-- MISS"
                print(f"    {key}: truth={truth} pred={pred} ({vid} {best:.3f}/{margin:.3f}){flag}")
        print(f"    confusion: {stats}")
        g1_ok = stats["G1"]["G1"] / max(1, sum(stats["G1"].values()))
        g2_ok = stats["G2"]["G2"] / max(1, sum(stats["G2"].values()))
        print(f"    G1 recall={g1_ok:.2f} G2 recall={g2_ok:.2f}")
        print("    PASS" if g1_ok >= 0.8 and g2_ok >= 0.8 else "    FAIL")
    except ImportError:
        print("    registry module unavailable; skipping")

    # 9. turn-06 regression fixture: long mixed turns must resolve as
    # segment sequences, and turn-level rollups must follow the policy.
    # Expected: 06 -> uncertain (split), 08 (owner) -> obey, media -> ack.
    print("\n[9] turn-level rollup regression (054936Z 06 / 052656Z 04):")
    checks = [
        ("20260928T054936Z/user_turn_06.wav", "uncertain"),
        ("20260928T054936Z/user_turn_08.wav", "obey"),
        ("20260928T052656Z/user_turn_04.wav", "ack"),
        ("20260928T052656Z/user_turn_01.wav", "obey"),
    ]
    ok = True
    for rel, want in checks:
        p = REPO / "data" / "sessions" / rel
        if not p.exists():
            print(f"    {rel}: MISSING")
            ok = False
            continue
        from registry import VoiceRegistry
        t = v.verify_turn(load16k(p), registry=VoiceRegistry())
        got = t["decision"]
        mark = "" if got == want else "  <-- MISMATCH"
        if got != want:
            ok = False
        segstr = ",".join(
            {"obey": "O", "ack": ".", "uncertain": "?"}.get(x, "?")
            for x in t.get("segment_votes", []))
        print(f"    {rel.split('/')[-1]}: {got} (want {want}) [{segstr}]{mark}")
    print("    PASS" if ok else "    FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

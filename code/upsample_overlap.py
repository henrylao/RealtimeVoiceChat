import base64
import logging
import os
import numpy as np
from scipy.signal import resample_poly
from typing import Optional

logger = logging.getLogger(__name__)

# -1 dBFS in linear amplitude: the ceiling wire peaks are limited to.
# (0 dB = loudest possible digital sample; -1 dB leaves sane headroom
# against downstream clipping while staying perceptually "full scale".)
CEILING_MINUS_1DB = 10.0 ** (-1.0 / 20.0)  # ≈ 0.8913


class UpsampleOverlap:
    """
    Manages chunk-wise audio upsampling with overlap handling.

    Wire contract: input chunks are mono PCM16 at SOURCE_RATE (24 kHz, the
    native rate of the Kokoro/Coqui/Orpheus TTS engines) and output chunks
    are mono PCM16 at TARGET_RATE (48 kHz). Browser playback MUST resample
    from TARGET_RATE to the actual AudioContext rate instead of playing
    samples 1:1, or Bluetooth routes at 44.1 kHz / 16 kHz sound slowed/flat.

    Loudness chain (all in 48 kHz float, before int16 encode), active only
    when TTS_LOUDNESS_ENABLED=true (default false = plain upsample, exactly
    the pre-boost behavior): fixed gain (+12 dB default) → optional
    broadcast compressor (threshold/ratio + makeup, with a downward gate so
    pauses stay silent) → peak limiter at the ceiling (-1 dBFS default).
    With loudness disabled, output is the straight resampled signal —
    no gain, no compression, no limiting.
    """

    SOURCE_RATE = 24000
    TARGET_RATE = 48000

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            logger.warning(f"Invalid {name} env var, using {default}")
            return default

    def __init__(
        self,
        gain_db: Optional[float] = None,
        ceiling_db: float = -1.0,
        comp_threshold_db: Optional[float] = None,
        comp_ratio: Optional[float] = None,
        comp_attack_ms: Optional[float] = None,
        comp_release_ms: Optional[float] = None,
        comp_makeup_db: Optional[float] = None,
        comp_gate_db: Optional[float] = None,
        comp_lookahead_ms: Optional[float] = None,
        comp_enabled: Optional[bool] = None,
        loudness_enabled: Optional[bool] = None,
    ):
        """
        Initializes the UpsampleOverlap processor.

        Sets up the internal state required for tracking previous audio chunks
        and their resampled versions to handle overlaps during processing.

        Loudness: fixed gain → compressor → peak limiter (see class docstring).
        Fixed gain (rather than per-chunk normalization) keeps loudness
        constant across chunks with no pumping. Calibrated from real Kokoro
        turns (peaks ≈ -5…-9 dBFS, RMS ≈ -26 dBFS).

        Args:
            gain_db: Fixed boost in dB. Defaults to env TTS_GAIN_DB or 12.0.
                     Set to 0 to disable the boost (limiter still applies).
            ceiling_db: Peak ceiling in dBFS. Defaults to env TTS_CEILING_DB
                        or -1.0. Must be <= 0.
            comp_threshold_db: Compressor threshold in dBFS, measured AFTER
                        the fixed gain. Env TTS_COMP_THRESHOLD_DB, def -24.0.
            comp_ratio: Compression ratio above threshold. Env TTS_COMP_RATIO,
                        default 3.0. Set to 1.0 to bypass compression (gate
                        and limiter still apply).
            comp_attack_ms: Envelope attack. Env TTS_COMP_ATTACK_MS, def 2.0.
            comp_release_ms: Envelope release. Env TTS_COMP_RELEASE_MS,
                        default 120.0.
            comp_makeup_db: Makeup gain applied after compression.
                        Env TTS_COMP_MAKEUP_DB, default 12.0.
            comp_gate_db: Below this envelope level, downward expansion keeps
                        pauses/noise floor down. Env TTS_COMP_GATE_DB,
                        default -50.0.
            comp_lookahead_ms: Lookahead delay so gain reduction anticipates
                        transients (no overshoot clipping). Env
                        TTS_COMP_LOOKAHEAD_MS, default 3.0.
            comp_enabled: Master switch for the compressor stage. Env
                        TTS_COMP_ENABLED, default false. When false, the
                        compressor, makeup gain and gate are bypassed
                        entirely. Set to true to enable broadcast-style
                        squashing. (TTS_COMP_RATIO=1.0 is an equivalent
                        manual bypass.) Only applies when loudness is enabled.
            loudness_enabled: Master switch for the whole loudness chain
                        (fixed gain + compressor + limiter). Env
                        TTS_LOUDNESS_ENABLED, default false. When false,
                        output is plain resampled audio with no gain, no
                        compression and no limiting — the original behavior.
        """
        self.previous_chunk: Optional[np.ndarray] = None
        self.resampled_previous_chunk: Optional[np.ndarray] = None
        if loudness_enabled is None:
            loudness_enabled = os.getenv("TTS_LOUDNESS_ENABLED", "false").lower() in ("1", "true", "yes")
        self.loudness_enabled = bool(loudness_enabled)
        if gain_db is None:
            gain_db = self._env_float("TTS_GAIN_DB", 12.0)
        ceiling_db = self._env_float("TTS_CEILING_DB", ceiling_db)
        ceiling_db = min(ceiling_db, 0.0)
        self.gain_db = gain_db
        self.gain_linear = 10.0 ** (gain_db / 20.0)
        self.ceiling = 10.0 ** (ceiling_db / 20.0)
        self.ceiling_db = ceiling_db
        # Soft-knee start 1 dB below the ceiling, 4:1 limiting above it,
        # hard clamp at the ceiling as the final guarantee. With overs of a
        # few dB (the realistic case after the fixed gain) this lands peaks
        # within ~1 dB of the ceiling instead of square-wave clipping them.
        self.knee = self.ceiling * (10.0 ** (-1.0 / 20.0))
        self.ratio = 4.0
        # Broadcast compressor state (envelope persists across chunks so
        # attack/release behave as one continuous detector per utterance).
        if comp_enabled is None:
            comp_enabled = os.getenv("TTS_COMP_ENABLED", "false").lower() in ("1", "true", "yes")
        self.comp_enabled = bool(comp_enabled)
        self.comp_thresh_db = (
            comp_threshold_db if comp_threshold_db is not None
            else self._env_float("TTS_COMP_THRESHOLD_DB", -24.0)
        )
        self.comp_ratio = (
            comp_ratio if comp_ratio is not None
            else self._env_float("TTS_COMP_RATIO", 3.0)
        )
        self.comp_ratio = max(self.comp_ratio, 1.0)
        attack_ms = (
            comp_attack_ms if comp_attack_ms is not None
            else self._env_float("TTS_COMP_ATTACK_MS", 2.0)
        )
        release_ms = (
            comp_release_ms if comp_release_ms is not None
            else self._env_float("TTS_COMP_RELEASE_MS", 120.0)
        )
        self.comp_makeup_db = (
            comp_makeup_db if comp_makeup_db is not None
            else self._env_float("TTS_COMP_MAKEUP_DB", 12.0)
        )
        self.comp_gate_db = (
            comp_gate_db if comp_gate_db is not None
            else self._env_float("TTS_COMP_GATE_DB", -50.0)
        )
        fs = float(self.TARGET_RATE)
        self._att_coeff = float(np.exp(-1.0 / (fs * max(attack_ms, 0.1) / 1000.0)))
        self._rel_coeff = float(np.exp(-1.0 / (fs * max(release_ms, 1.0) / 1000.0)))
        self._comp_env = 0.0
        lookahead_ms = (
            comp_lookahead_ms if comp_lookahead_ms is not None
            else self._env_float("TTS_COMP_LOOKAHEAD_MS", 3.0)
        )
        self._lookahead_n = int(round(fs * max(lookahead_ms, 0.0) / 1000.0))
        self._dl_tail = np.zeros(self._lookahead_n, dtype=np.float64)
        if not self.loudness_enabled:
            logger.info("UpsampleOverlap: loudness chain disabled (plain resample, no gain/limiting)")
        elif self.comp_enabled:
            logger.info(
                f"UpsampleOverlap: gain +{self.gain_db:.1f} dB, "
                f"comp {self.comp_ratio:.1f}:1 @ {self.comp_thresh_db:.0f} dBFS "
                f"(att {attack_ms:.0f}ms / rel {release_ms:.0f}ms, "
                f"lookahead {lookahead_ms:.0f}ms, "
                f"makeup +{self.comp_makeup_db:.1f} dB, gate {self.comp_gate_db:.0f} dBFS), "
                f"ceiling {ceiling_db:.1f} dBFS"
            )
        else:
            logger.info(
                f"UpsampleOverlap: gain +{self.gain_db:.1f} dB, "
                f"comp bypassed (plain gain + limiter), "
                f"ceiling {ceiling_db:.1f} dBFS"
            )

    def _apply_compressor(self, x: np.ndarray) -> np.ndarray:
        """
        Broadcast-style compression with lookahead, makeup gain and gate.

        A peak envelope follower (attack/release, state carried across chunks
        so each utterance sees one continuous detector) drives the gain
        computer: ratio compression above threshold, plus makeup. The computed
        gain is applied to audio delayed by the lookahead window, so the gain
        reduction is already in place when a transient sounds — no overshoot
        clipping, no onset clicks. Below the gate level, downward expansion
        keeps pauses and the synth noise floor from being lifted into hiss.
        Returns the compressed signal; peak safety is left to the limiter
        stage downstream. When the compressor is disabled
        (TTS_COMP_ENABLED=false) this is a pure bypass: input passes
        through untouched.
        """
        if x.size == 0 or not self.comp_enabled or self.comp_ratio <= 1.0:
            return x
        n = x.shape[0]
        n_look = self._lookahead_n
        # Detector runs on current input; gain applies to delayed audio.
        xn = np.concatenate((self._dl_tail, x)) if n_look else x
        ax = np.abs(xn).astype(np.float64, copy=False)
        env = np.empty_like(ax)
        e = self._comp_env
        a_att, a_rel = self._att_coeff, self._rel_coeff
        for i, v in enumerate(ax):
            c = a_att if v > e else a_rel
            e = c * e + (1.0 - c) * v
            env[i] = e
        self._comp_env = float(e)
        eps = 1e-9
        env_db = 20.0 * np.log10(np.maximum(env, eps))
        over = np.maximum(env_db - self.comp_thresh_db, 0.0)
        comp_gr = over * (1.0 - 1.0 / self.comp_ratio)
        below = np.maximum(self.comp_gate_db - env_db, 0.0)
        gate_gr = np.minimum(below * 0.5, 30.0)
        total_db = self.comp_makeup_db - comp_gr - gate_gr
        g = 10.0 ** (total_db / 20.0)
        if n_look:
            # First call: delay line prefilled with zeros; drain it now.
            out = xn[:n] * g[n_look:n_look + n]
            self._dl_tail = xn[n:]
        else:
            out = xn * g
        return out

    def _drain_compressor(self) -> np.ndarray:
        """Flushes the lookahead delay line with its current decayed gain."""
        tail = self._dl_tail
        self._dl_tail = np.zeros(self._lookahead_n, dtype=np.float64)
        if tail.size == 0:
            return tail
        e = self._comp_env
        eps = 1e-9
        env_db = 20.0 * np.log10(max(e, eps))
        over = max(env_db - self.comp_thresh_db, 0.0)
        comp_gr = over * (1.0 - 1.0 / self.comp_ratio)
        below = max(self.comp_gate_db - env_db, 0.0)
        gate_gr = min(below * 0.5, 30.0)
        return tail * (10.0 ** ((self.comp_makeup_db - comp_gr - gate_gr) / 20.0))

    def _apply_gain_and_limit(self, part: np.ndarray) -> np.ndarray:
        """
        Full loudness chain: fixed gain → compressor → peak limiter.

        Linear below the knee (transparent for the bulk of speech), 4:1
        soft limiting above the knee (rounds off hot peaks instead of
        square-wave clipping them), and a final hard clamp so wire peaks
        can never exceed the ceiling (default -1 dBFS).

        When loudness is disabled (TTS_LOUDNESS_ENABLED=false), the input
        passes through untouched: plain resampled audio, no gain, no
        compression, no limiting.
        """
        if part.size == 0 or not self.loudness_enabled:
            return part
        x = self._apply_compressor(part * self.gain_linear)
        ax = np.abs(x)
        over = ax > self.knee
        if np.any(over):
            # Work in dB relative to full scale for the compression curve.
            eps = 1e-9
            x_db = 20.0 * np.log10(np.maximum(ax, eps))
            knee_db = 20.0 * np.log10(max(self.knee, eps))
            ceil_db = 20.0 * np.log10(max(self.ceiling, eps))
            y_db = np.where(
                x_db <= knee_db,
                x_db,
                knee_db + (x_db - knee_db) / self.ratio,
            )
            y_db = np.minimum(y_db, ceil_db)
            x = np.sign(x) * (10.0 ** (y_db / 20.0))
        # Final guarantee: nothing leaves hotter than the ceiling.
        return np.clip(x, -self.ceiling, self.ceiling)

    def get_base64_chunk(self, chunk: bytes) -> str:
        """
        Processes an incoming audio chunk, upsamples it, and returns the relevant segment as Base64.

        Converts the raw PCM bytes (assumed 16-bit signed integer) chunk to a
        float32 numpy array, scales it to [-1.0, 1.0], and upsamples from 24kHz to 48kHz.
        It uses the previous chunk's data to create an overlap, resamples the
        combined audio, and extracts the central portion corresponding primarily
        to the current chunk, using overlap to smooth transitions. The state is
        updated for the next call. The extracted audio segment is converted back
        to 16-bit PCM bytes and returned as a Base64 encoded string.

        Args:
            chunk: Raw audio data bytes (PCM 16-bit signed integer format expected).

        Returns:
            A Base64 encoded string representing the upsampled audio segment
            corresponding to the input chunk, adjusted for overlap. Returns an
            empty string if the input chunk is empty.
        """
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        # Handle potential empty chunks gracefully
        if audio_int16.size == 0:
             return "" # Return empty string for empty input chunk

        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Upsample the current chunk independently first, needed for state and first chunk logic
        upsampled_current_chunk = resample_poly(audio_float, self.TARGET_RATE, self.SOURCE_RATE)

        if self.previous_chunk is None:
            # First chunk: Output the first half of its upsampled version
            half = len(upsampled_current_chunk) // 2
            part = upsampled_current_chunk[:half]
        else:
            # Subsequent chunks: Combine previous float chunk with current float chunk
            combined = np.concatenate((self.previous_chunk, audio_float))
            # Upsample the combined chunk
            up = resample_poly(combined, self.TARGET_RATE, self.SOURCE_RATE)

            # Calculate lengths and indices for extracting the middle part
            # Ensure self.resampled_previous_chunk is not None (shouldn't happen here due to outer if)
            assert self.resampled_previous_chunk is not None
            prev_len = len(self.resampled_previous_chunk) # Length of the *upsampled* previous chunk
            h_prev = prev_len // 2 # Midpoint index of the *upsampled* previous chunk

            # *** CORRECTED INDEX CALCULATION (Reverted to original) ***
            # Calculate the end index for the part corresponding to the current chunk's main contribution
            # This index represents the midpoint of the *current* chunk's contribution within the combined 'up' array.
            h_cur = (len(up) - prev_len) // 2 + prev_len

            part = up[h_prev:h_cur]

        # Update state for the next iteration
        self.previous_chunk = audio_float
        self.resampled_previous_chunk = upsampled_current_chunk # Store the upsampled *current* chunk for the *next* overlap

        # Convert the extracted part back to PCM16 bytes and encode
        pcm = (self._apply_gain_and_limit(part) * 32767).astype(np.int16).tobytes()
        return base64.b64encode(pcm).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        """
        Returns the final remaining segment of upsampled audio after all chunks are processed.

        After the last call to `get_base64_chunk`, the state holds the upsampled
        version of the very last input chunk (`self.resampled_previous_chunk`).
        This method returns that *entire* final upsampled chunk, converted to
        16-bit PCM bytes and encoded as Base64. It then clears the internal state.
        This should be called once after all input chunks have been passed to `get_base64_chunk`.

        Returns:
            A Base64 encoded string containing the final upsampled audio chunk,
            or None if no chunks were processed or if flush has already been called.
        """
        # *** CORRECTED FLUSH LOGIC (Reverted to original) ***
        if self.resampled_previous_chunk is not None:
            # Return the entire last upsampled chunk as per original logic
            pcm = (self._apply_gain_and_limit(self.resampled_previous_chunk) * 32767).astype(np.int16).tobytes()

            # Clear state after flushing
            self.previous_chunk = None
            self.resampled_previous_chunk = None
            return base64.b64encode(pcm).decode('utf-8')
        return None # Return None if there's nothing to flush
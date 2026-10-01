import asyncio
import logging
import os
import time
from collections import deque
from typing import Optional, Callable, Deque, Tuple
import numpy as np
from scipy.signal import resample_poly
from transcribe import TranscriptionProcessor

# Optional live speaker scoring. The speaker_id/ prototype stays standalone;
# run.sh adds it to PYTHONPATH. Absent/unavailable -> silently disabled.
try:
    from speaker_verify import get_verifier
    _SPEAKER_VERIFY_IMPORT_OK = True
except ImportError:
    get_verifier = None  # type: ignore
    _SPEAKER_VERIFY_IMPORT_OK = False

logger = logging.getLogger(__name__)


class AudioInputProcessor:
    """
    Manages audio input, processes it for transcription, and handles related callbacks.

    This class receives raw audio chunks, resamples them to the required format (16kHz),
    feeds them to an underlying `TranscriptionProcessor`, and manages callbacks for
    real-time transcription updates, recording start events, and silence detection.
    It also runs the transcription process in a background task.
    """

    _DEFAULT_SOURCE_RATE = 48000  # Historical assumption when clients signal nothing.
    _TARGET_RATE = 16000

    def __init__(
            self,
            language: str = "en",
            is_orpheus: bool = False,
            silence_active_callback: Optional[Callable[[bool], None]] = None,
            pipeline_latency: float = 0.5,
        ) -> None:
        """
        Initializes the AudioInputProcessor.

        Args:
            language: Target language code for transcription (e.g., "en").
            is_orpheus: Flag indicating if a specific model variant should be used.
            silence_active_callback: Optional callback function invoked when silence state changes.
                                     It receives a boolean argument (True if silence is active).
            pipeline_latency: Estimated latency of the processing pipeline in seconds.
        """
        self.last_partial_text: Optional[str] = None
        self.transcriber = TranscriptionProcessor(
            language,
            on_recording_start_callback=self._on_recording_start,
            silence_active_callback=self._silence_active_callback,
            is_orpheus=is_orpheus,
            pipeline_latency=pipeline_latency,
        )
        # Flag to indicate if the transcription loop has failed fatally
        self._transcription_failed = False
        self.transcription_task = asyncio.create_task(self._run_transcription_loop())


        self.realtime_callback: Optional[Callable[[str], None]] = None
        self.recording_start_callback: Optional[Callable[[None], None]] = None # Type adjusted
        self.silence_active_callback: Optional[Callable[[bool], None]] = silence_active_callback
        self.interrupted = False # TODO: Consider renaming or clarifying usage (interrupted by user speech?)

        # Watchdog state: distinguishes "no audio arriving" from "audio
        # arriving but untranscribed" (deaf-but-green). Updated on the hot
        # path with cheap aggregates only; read by stt_status()/health.
        self._fed_rms: float = 0.0          # decaying RMS of fed audio (0..1)
        self._last_feed_time: float = 0.0   # last fed chunk timestamp
        self.last_partial_time: float = 0.0  # last realtime partial timestamp
        self.STT_STALL_AFTER_S = 45.0       # no partial this long => stalled
        self.STT_SOUND_RMS = 0.005          # fed audio above this counts as sound
        self.STT_FLOW_RECENT_S = 15.0       # fed audio this recent counts as flowing

        # Last client capture rate seen (Hz). Exposed for per-turn manifest
        # attribution (e.g. laptop 48k vs iPhone 44.1k mics).
        self.last_sample_rate: int = self._DEFAULT_SOURCE_RATE

        # Flag-only live speaker scoring (observability; never gates audio).
        self.speaker_buffer = np.zeros(0, dtype=np.int16)  # accumulated 16kHz PCM
        self.last_speaker_score: Optional[dict] = None
        self._last_speaker_check = 0.0
        self._speaker_check_interval = 4.0  # seconds between background checks
        self._speaker_window_s = 3.0  # seconds of audio per verification window

        # Sustained-voicing accumulator for the barge-in fusion gate.
        # Every fed chunk is scored voiced/unvoiced (speech-like structure,
        # not just energy) and logged with a timestamp; barge confirmation
        # sums voiced milliseconds since the episode started. Coughs, thumps
        # and pops are loud but brief and spectrally flat, so they barely
        # accumulate; real speech piles up hundreds of ms within a second.
        # Calibrated on saved mic audio (speech: dense + low ZCR; coughs:
        # sparse + high ZCR).
        def _env_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except ValueError:
                logger.warning(f"👂⚠️ Invalid {name} env var. Using default: {default}")
                return default
        self.VOICED_RMS = _env_float("VOICED_RMS", 0.02)
        self.VOICED_ZCR = _env_float("VOICED_ZCR", 0.13)
        self._voiced_log: Deque[Tuple[float, float]] = deque(maxlen=2048)
        self._voiced_window_s = 10.0

        self._setup_callbacks()
        logger.info("👂🚀 AudioInputProcessor initialized.")

    def _silence_active_callback(self, is_active: bool) -> None:
        """Internal callback relay for silence detection status."""
        if self.silence_active_callback:
            self.silence_active_callback(is_active)

    def _on_recording_start(self) -> None:
        """Internal callback relay triggered when the transcriber starts recording."""
        if self.recording_start_callback:
            self.recording_start_callback()

    def abort_generation(self) -> None:
        """Signals the underlying transcriber to abort any ongoing generation process."""
        logger.info("👂🛑 Aborting generation requested.")
        self.transcriber.abort_generation()

    def _setup_callbacks(self) -> None:
        """Sets up internal callbacks for the TranscriptionProcessor instance."""
        def partial_transcript_callback(text: str) -> None:
            """Handles partial transcription results from the transcriber."""
            self.last_partial_time = time.time()
            if text != self.last_partial_text:
                self.last_partial_text = text
                if self.realtime_callback:
                    self.realtime_callback(text)

        self.transcriber.realtime_transcription_callback = partial_transcript_callback

    def voiced_ms_since(self, t0: float) -> float:
        """Sums voiced milliseconds fed at/after timestamp t0.

        Used by the barge-in fusion gate: confirmation requires sustained
        speech-like audio, not just words STT invented from a thump. Prunes
        entries older than the window. Never raises.
        """
        try:
            cutoff = time.time() - self._voiced_window_s
            total = 0.0
            # Deque appends on the right; iterate from newest for early exit.
            for ts, ms in reversed(self._voiced_log):
                if ts < t0:
                    break
                total += ms
            # Opportunistic prune of the old end.
            while self._voiced_log and self._voiced_log[0][0] < cutoff:
                self._voiced_log.popleft()
            return total
        except Exception:
            return 0.0

    def _score_chunk_voiced(self, processed: np.ndarray) -> float:
        """Scores one resampled (16 kHz int16) chunk, returning its voiced
        milliseconds (chunk duration if speech-like, else 0.0).

        Voiced = energy above VOICED_RMS with a low zero-crossing rate
        (harmonic structure). Broadband bursts (thumps, pops) and flat room
        noise fail the ZCR side even when loud.
        """
        try:
            n = processed.size
            if n == 0:
                return 0.0
            x = processed.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(x * x)))
            if rms < self.VOICED_RMS:
                return 0.0
            signs = np.sign(x)
            # Crossings per sample (matches the calibration analysis scale:
            # voiced speech ≈ 0.05-0.10, bursts/noise ≈ 0.15+).
            zcr = float(np.count_nonzero(signs[1:] != signs[:-1]) / max(n - 1, 1))
            if zcr > self.VOICED_ZCR:
                return 0.0
            return n / 16000.0 * 1000.0
        except Exception:
            return 0.0

    def stt_status(self) -> dict:
        """Snapshot for watchdog/health: is audio flowing unheard?

        stalled=True means sound-level audio has been fed recently with no
        partial transcription for a long while while the backend claims to
        be alive — i.e. deaf-but-green. Never raises.
        """
        try:
            now = time.time()
            flowing = (now - self._last_feed_time) < self.STT_FLOW_RECENT_S
            sounding = self._fed_rms > self.STT_SOUND_RMS
            quiet_long = (now - self.last_partial_time) > self.STT_STALL_AFTER_S
            stalled = (flowing and sounding and quiet_long
                       and not self._transcription_failed)
            return {"fed_rms": round(self._fed_rms, 5),
                    "secs_since_partial": round(now - self.last_partial_time, 1)
                    if self.last_partial_time else None,
                    "stalled": stalled,
                    "backend_failed": self._transcription_failed}
        except Exception:
            return {"fed_rms": 0.0, "secs_since_partial": None,
                    "stalled": False, "backend_failed": False}

    async def _run_transcription_loop(self) -> None:
        """
        Continuously runs the transcription loop in a background asyncio task.

        It repeatedly calls the underlying `transcribe_loop`. Per-cycle errors
        (e.g. a single bad/empty clip like faster-whisper's "No clip timestamps
        found") are treated as transient: logged with a short backoff, then the
        loop continues. Only after MAX_CONSECUTIVE_FAILURES in a row is the
        backend considered dead and `_transcription_failed` set. Handles
        CancelledError separately.
        """
        MAX_CONSECUTIVE_FAILURES = 5
        ERROR_BACKOFF_S = 1.0
        task_name = self.transcription_task.get_name() if hasattr(self.transcription_task, 'get_name') else 'TranscriptionTask'
        logger.info(f"👂▶️ Starting background transcription task ({task_name}).")
        consecutive_failures = 0
        while True: # Loop restored to continuously call transcribe_loop
            try:
                # Run one cycle of the underlying blocking loop
                await asyncio.to_thread(self.transcriber.transcribe_loop)
                # If transcribe_loop returns without error, it means one cycle is complete.
                # The `while True` ensures it will be called again.
                logger.debug("👂✅ TranscriptionProcessor.transcribe_loop completed one cycle.")
                consecutive_failures = 0 # A clean cycle resets the failure streak
                # Add a small sleep to prevent potential tight loop if transcribe_loop returns instantly
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                logger.info(f"👂🚫 Transcription loop ({task_name}) cancelled.")
                # Do not set failure flag on cancellation
                break # Exit the while loop
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(f"👂💥 Transcription loop ({task_name}) failed {consecutive_failures}x in a row, giving up: {e}. Loop terminated.", exc_info=True)
                    self._transcription_failed = True # Set failure flag
                    break # Exit the while loop, stopping retries
                logger.warning(f"👂⚠️ Transcription loop ({task_name}) transient error ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}. Backing off {ERROR_BACKOFF_S}s and continuing.")
                await asyncio.sleep(ERROR_BACKOFF_S)

        logger.info(f"👂⏹️ Background transcription task ({task_name}) finished.")


    def process_audio_chunk(self, raw_bytes: bytes, source_rate: int = _DEFAULT_SOURCE_RATE) -> np.ndarray:
        """
        Converts raw audio bytes (int16) at `source_rate` to 16kHz 16-bit PCM.

        The audio is converted to float32 for accurate resampling and then
        converted back to int16, clipping values outside the valid range.
        Clients report their actual capture rate (iOS ignores requested rates);
        unknown/invalid rates fall back to the historical 48 kHz assumption.

        Args:
            raw_bytes: Raw audio data assumed to be in int16 format.
            source_rate: Capture sample rate in Hz of the incoming audio.

        Returns:
            A numpy array containing the resampled audio in int16 format at 16kHz.
            Returns an array of zeros if the input is silent.
        """
        if not isinstance(source_rate, int) or source_rate <= 0:
            source_rate = self._DEFAULT_SOURCE_RATE
        raw_audio = np.frombuffer(raw_bytes, dtype=np.int16)

        if np.max(np.abs(raw_audio)) == 0:
            # Calculate expected length after resampling for silence
            expected_len = int(np.ceil(len(raw_audio) * self._TARGET_RATE / source_rate))
            return np.zeros(expected_len, dtype=np.int16)

        # Convert to float32 for resampling precision
        audio_float32 = raw_audio.astype(np.float32)

        # Dynamic resampling ratio (e.g. 44100 -> 16000 is up=160, down=441)
        import math
        div = math.gcd(source_rate, self._TARGET_RATE)
        resampled_float = resample_poly(audio_float32,
                                        self._TARGET_RATE // div,
                                        source_rate // div)

        # Convert back to int16, clipping to ensure validity
        resampled_int16 = np.clip(resampled_float, -32768, 32767).astype(np.int16)

        return resampled_int16


    def _accumulate_speaker_audio(self, processed: np.ndarray) -> None:
        """Accumulate resampled PCM for flag-only speaker verification.

        Fires a background check every few seconds once a full window is
        buffered. Observability only: results are logged and stored on
        `self.last_speaker_score`; audio flow is never modified.
        """
        if not _SPEAKER_VERIFY_IMPORT_OK:
            return
        try:
            self.speaker_buffer = np.concatenate([self.speaker_buffer, processed])
            max_len = int(16000 * (self._speaker_window_s + 1.0))
            if len(self.speaker_buffer) > max_len:
                self.speaker_buffer = self.speaker_buffer[-max_len:]
            now = time.time()
            need_len = int(16000 * self._speaker_window_s)
            if (len(self.speaker_buffer) >= need_len
                    and now - self._last_speaker_check >= self._speaker_check_interval):
                self._last_speaker_check = now
                window = self.speaker_buffer[-need_len:].copy()
                asyncio.create_task(self._run_speaker_check(window))
        except Exception as e:
            logger.debug(f"🔐 speaker accumulation skipped: {e}")

    async def _run_speaker_check(self, window: np.ndarray) -> None:
        """Run one verification window in a thread; log the score."""
        try:
            verifier = get_verifier() if get_verifier else None
            if verifier is None:
                return
            result = await asyncio.to_thread(verifier.verify_window, window)
            self.last_speaker_score = result
            if result.get("reliable"):
                logger.info(
                    f"🔐 speaker score={result['speaker_score']} "
                    f"(anchor={result.get('score_anchor')}) "
                    f"owner={result['is_owner']} window={result['window_s']}s"
                )
        except Exception as e:
            logger.debug(f"🔐 speaker check skipped: {e}")

    async def process_chunk_queue(self, audio_queue: asyncio.Queue) -> None:
        """
        Continuously processes audio chunks received from an asyncio Queue.

        Retrieves audio data, processes it using `process_audio_chunk`, and
        feeds the result to the transcriber unless interrupted or the transcription
        task has failed. Stops when `None` is received from the queue or upon error.

        Args:
            audio_queue: An asyncio queue expected to yield dictionaries containing
                         'pcm' (raw audio bytes) or None to terminate.
        """
        logger.info("👂▶️ Starting audio chunk processing loop.")
        while True:
            try:
                # Check if the transcription task has permanently failed *before* getting item
                if self._transcription_failed:
                    logger.error("👂🛑 Transcription task failed previously. Stopping audio processing.")
                    break # Stop processing if transcription backend is down

                # Check if the task finished unexpectedly (e.g., cancelled but not failed)
                # Needs to check self.transcription_task existence as it might be None during shutdown
                if self.transcription_task and self.transcription_task.done() and not self._transcription_failed:
                     # Attempt to check exception status if task is done
                    task_exception = self.transcription_task.exception()
                    if task_exception and not isinstance(task_exception, asyncio.CancelledError):
                        # If there was an exception other than CancelledError, treat it as failed.
                        logger.error(f"👂🛑 Transcription task finished with unexpected error: {task_exception}. Stopping audio processing.", exc_info=task_exception)
                        self._transcription_failed = True # Mark as failed
                        break
                    else:
                         # Finished cleanly or was cancelled
                        logger.warning("👂⏹️ Transcription task is no longer running (completed or cancelled). Stopping audio processing.")
                        break # Stop processing

                audio_data = await audio_queue.get()
                if audio_data is None:
                    logger.info("👂🔌 Received termination signal for audio processing.")
                    break  # Termination signal

                pcm_data = audio_data.pop("pcm")

                # Process audio chunk (resampling from the client-signaled rate)
                source_rate = audio_data.get("sample_rate", self._DEFAULT_SOURCE_RATE)
                self.last_sample_rate = source_rate
                processed = self.process_audio_chunk(pcm_data, source_rate)
                if processed.size == 0:
                    continue # Skip empty chunks

                # Watchdog feed gauge (cheap decaying RMS of what STT receives).
                try:
                    chunk_rms = float(np.sqrt(np.mean(
                        processed.astype(np.float32) ** 2))) / 32768.0
                    self._fed_rms = 0.95 * self._fed_rms + 0.05 * chunk_rms
                    self._last_feed_time = time.time()
                except Exception:
                    pass

                # Sustained-voicing log for the barge-in fusion gate.
                try:
                    now = time.time()
                    self._voiced_log.append((now, self._score_chunk_voiced(processed)))
                    cutoff = now - self._voiced_window_s
                    while self._voiced_log and self._voiced_log[0][0] < cutoff:
                        self._voiced_log.popleft()
                except Exception:
                    pass

                # Flag-only speaker verification: accumulate 16kHz PCM and score
                # in the background every few seconds. Never gates or alters audio.
                self._accumulate_speaker_audio(processed)

                # Feed audio only if not interrupted and transcriber should be running
                if not self.interrupted:
                    # Check failure flag again, as it might have been set between queue.get and here
                     if not self._transcription_failed:
                        # Feed audio to the underlying processor
                        self.transcriber.feed_audio(processed.tobytes(), audio_data)
                     # No 'else' needed here because the checks at the start of the loop handle termination
                else:
                    # Muted globally (end-of-turn or stale). Count so logs show
                    # server-side drops vs client sending nothing.
                    self._interrupted_drops = getattr(self, "_interrupted_drops", 0) + 1
                    if self._interrupted_drops == 1 or self._interrupted_drops % 500 == 0:
                        logger.info(f"👂⏸️ Dropping mic chunks while interrupted ({self._interrupted_drops} this processor lifetime)")

            except asyncio.CancelledError:
                logger.info("👂🚫 Audio processing task cancelled.")
                break
            except Exception as e:
                # Log general errors during audio chunk processing
                logger.error(f"👂💥 Audio processing error in queue loop: {e}", exc_info=True)
                # Continue processing subsequent chunks after logging the error.
                # Consider adding logic to break if errors persist.
        logger.info("👂⏹️ Audio chunk processing loop finished.")


    def shutdown(self) -> None:
        """
        Initiates shutdown procedures for the audio processor and transcriber.

        Signals the transcriber to shut down and cancels the background
        transcription task.
        """
        logger.info("👂🛑 Shutting down AudioInputProcessor...")
        # Ensure transcriber shutdown is called first to signal the loop
        if hasattr(self.transcriber, 'shutdown'):
             logger.info("👂🛑 Signaling TranscriptionProcessor to shut down.")
             self.transcriber.shutdown()
        else:
             logger.warning("👂⚠️ TranscriptionProcessor does not have a shutdown method.")

        if self.transcription_task and not self.transcription_task.done():
            task_name = self.transcription_task.get_name() if hasattr(self.transcription_task, 'get_name') else 'TranscriptionTask'
            logger.info(f"👂🚫 Cancelling background transcription task ({task_name})...")
            self.transcription_task.cancel()
            # Optional: Add await with timeout here in an async shutdown context
            # try:
            #     await asyncio.wait_for(self.transcription_task, timeout=5.0)
            # except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
            #     logger.warning(f"👂⚠️ Error/Timeout waiting for transcription task {task_name} cancellation: {e}")
        else:
            logger.info("👂✅ Transcription task already done or not running during shutdown.")

        logger.info("👂👋 AudioInputProcessor shutdown sequence initiated.")
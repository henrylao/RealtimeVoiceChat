# server.py
from queue import Queue, Empty
import logging
from collections import deque
from difflib import SequenceMatcher
from logsetup import setup_logging
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    logger.info("🖥️👋 Welcome to local real-time voice chat")

from upsample_overlap import UpsampleOverlap
from datetime import datetime
from colors import Colors
import uvicorn
import asyncio
import struct
import json
import time
import threading # Keep threading for SpeechPipelineManager internals and AbortWorker
import sys
import os # Added for environment variable access

from typing import Any, Dict, Optional, Callable # Added for type hints in docstrings
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, Response, FileResponse

# Native serve: external Ollama endpoint + Kokoro TTS by default.
# All tunables are env-driven (engines stay selectable as config).
USE_SSL = os.getenv("USE_SSL", "false").lower() in ("1", "true", "yes")
TTS_START_ENGINE = os.getenv("TTS_ENGINE", "kokoro")
TTS_ORPHEUS_MODEL = os.getenv(
    "TTS_ORPHEUS_MODEL",
    "orpheus-3b-0.1-ft-Q8_0-GGUF/orpheus-3b-0.1-ft-q8_0.gguf",
)

LLM_START_PROVIDER = os.getenv("LLM_PROVIDER", "llamacpp")
LLM_START_MODEL = os.getenv("LLM_MODEL", "gemma-4-12B-it-qat-UD-Q4_K_XL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8081/v1")
NO_THINK = os.getenv("NO_THINK", "false").lower() in ("1", "true", "yes")
DIRECT_STREAM = TTS_START_ENGINE=="orpheus"

if __name__ == "__main__":
    logger.info(f"🖥️⚙️ {Colors.apply('[PARAM]').blue} Starting engine: {Colors.apply(TTS_START_ENGINE).blue}")
    logger.info(f"🖥️⚙️ {Colors.apply('[PARAM]').blue} Direct streaming: {Colors.apply('ON' if DIRECT_STREAM else 'OFF').blue}")

# Define the maximum allowed size for the incoming audio queue
try:
    MAX_AUDIO_QUEUE_SIZE = int(os.getenv("MAX_AUDIO_QUEUE_SIZE", 50))
    if __name__ == "__main__":
        logger.info(f"🖥️⚙️ {Colors.apply('[PARAM]').blue} Audio queue size limit set to: {Colors.apply(str(MAX_AUDIO_QUEUE_SIZE)).blue}")
except ValueError:
    if __name__ == "__main__":
        logger.warning("🖥️⚠️ Invalid MAX_AUDIO_QUEUE_SIZE env var. Using default: 50")
    MAX_AUDIO_QUEUE_SIZE = 50

# Self-interrupt (echo barge-in) guard tunables. The mic hears the assistant's
# own speaker output on loud routes; these distinguish echo from real speech
# without delaying genuine barge-ins (no debounce timers anywhere).
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning(f"🖥️⚠️ Invalid {name} env var. Using default: {default}")
        return default

# After the client reports TTS stop, still treat mic input as echo-suspect
# for this long (room/speaker tail). Genuine speech here loses nothing: with
# no generation running there is nothing to abort, it just starts a turn.
TTS_ECHO_HANGOVER_S = _env_float("TTS_ECHO_HANGOVER_S", 0.5)
# Assistant text produced within [MIN_DELAY, WINDOW] seconds ago counts as
# echo-matchable: recent enough to be in the air, old enough to have played.
TTS_ECHO_WINDOW_S = _env_float("TTS_ECHO_WINDOW_S", 4.0)
TTS_ECHO_MIN_DELAY_S = _env_float("TTS_ECHO_MIN_DELAY_S", 0.3)
# Consecutive suppressed partials before one is let through anyway (safety
# valve so a pathological match can't mute the user forever).
TTS_ECHO_MAX_SUPPRESS = int(_env_float("TTS_ECHO_MAX_SUPPRESS", 6))
TTS_ECHO_ENABLED = os.getenv("TTS_ECHO_GUARD", "true").lower() in ("1", "true", "yes")

# Stern barge-in: an armed episode confirms only on sustained speech-like
# audio (voiced milliseconds accumulated since arming) AND non-echo words.
# Words alone never confirm (STT invents them from thumps); audio alone
# never confirms (echo is audio too). Both legs name their verdict in audit.
# Lower for snappier, raise for sterner. 0 disables the audio leg (words +
# echo-match only, previous behavior). Default 500 ms: real speech in the
# saved corpus accumulates ~750+ ms; coughs/throat-clearing stall below
# ~400 ms.
TTS_BARGE_MIN_VOICED_MS = _env_float("TTS_BARGE_MIN_VOICED_MS", 500.0)

# TTS chunk pacing: deliver audio at ~realtime so the client's buffer hovers
# around a second instead of accumulating whole answers ahead of playback
# (long tails decouple server generations from what the user actually hears,
# which breaks barge-in). Multiplier on chunk audio seconds; 0 disables.
TTS_PACE_RT = _env_float("TTS_PACE_RT", 1.0)
# Preempt stale client playback when a genuine new turn starts speaking.
TTS_PREEMPT_ENABLED = os.getenv("TTS_PREEMPT_ENABLED", "true").lower() in ("1", "true", "yes")
# Stuck-playback watchdog: warn if the client still reports playing this long
# after tts_end with no new audio flowing (latch-True flag or wedged client).
TTS_STUCK_WARN_S = _env_float("TTS_STUCK_WARN_S", 30.0)

# Re-emission suppression: a final whose words imply an un-speakable rate
# (words per second over the VAD episode) is a stale STT buffer finalized
# twice, not fresh speech — drop it as a turn. Humans sustain ~2-4 wps;
# 5.0 leaves headroom (episode duration includes trailing silence, so real
# rates read low). Exact duplicates at plausible rates (frustrated repeats)
# still pass; only physics violations are dropped.
TTS_TURN_MAX_WPS = _env_float("TTS_TURN_MAX_WPS", 5.0)


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#from handlerequests import LanguageProcessor
#from audio_out import AudioOutProcessor
from audio_in import AudioInputProcessor
from speech_pipeline_manager import SpeechPipelineManager
from audio_dump import get_dumper
from colors import Colors

# Absolute paths so the server works regardless of CWD (repo root or code/).
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

LANGUAGE = "en"
# TTS_FINAL_TIMEOUT = 0.5 # unsure if 1.0 is needed for stability
TTS_FINAL_TIMEOUT = 1.0 # unsure if 1.0 is needed for stability

# --------------------------------------------------------------------
# Custom no-cache StaticFiles
# --------------------------------------------------------------------
class NoCacheStaticFiles(StaticFiles):
    """
    Serves static files without allowing client-side caching.

    Overrides the default Starlette StaticFiles to add 'Cache-Control' headers
    that prevent browsers from caching static assets. Useful for development.
    """
    async def get_response(self, path: str, scope: Dict[str, Any]) -> Response:
        """
        Gets the response for a requested path, adding no-cache headers.

        Args:
            path: The path to the static file requested.
            scope: The ASGI scope dictionary for the request.

        Returns:
            A Starlette Response object with cache-control headers modified.
        """
        response: Response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        # These might not be strictly necessary with no-store, but belt and suspenders
        if "etag" in response.headers:
             response.headers.__delitem__("etag")
        if "last-modified" in response.headers:
             response.headers.__delitem__("last-modified")
        return response

# --------------------------------------------------------------------
# Lifespan management
# --------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan, initializing and shutting down resources.

    Initializes global components like SpeechPipelineManager, Upsampler, and
    AudioInputProcessor and stores them in `app.state`. Handles cleanup on shutdown.

    Args:
        app: The FastAPI application instance.
    """
    logger.info("🖥️▶️ Server starting up")
    logger.info(f"🖥️⚙️ TTS engine: {TTS_START_ENGINE}, LLM: {LLM_START_PROVIDER}/{LLM_START_MODEL}")
    # Fail fast when the external LLM backend is down or the model is missing.
    if LLM_START_PROVIDER in ("ollama", "llamacpp", "lmstudio", "openai"):
        from check_backend import check_backend
        try:
            base = {
                "ollama": OLLAMA_BASE_URL,
                "llamacpp": LLAMA_BASE_URL,
            }.get(LLM_START_PROVIDER)
            models = check_backend(LLM_START_PROVIDER, base, LLM_START_MODEL)
            logger.info(f"🖥️⚙️ {LLM_START_PROVIDER} models available: {models}")
        except RuntimeError as e:
            logger.error(f"🖥️💥 {e}")
            raise
    # Opt-in per-turn audio collection (no-op unless AUDIO_DUMP=true).
    get_dumper(config={
        "tts_engine": TTS_START_ENGINE,
        "llm_provider": LLM_START_PROVIDER,
        "llm_model": LLM_START_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "llama_base_url": LLAMA_BASE_URL,
        "language": LANGUAGE,
    })
    # Initialize global components, not connection-specific state
    app.state.SpeechPipelineManager = SpeechPipelineManager(
        tts_engine=TTS_START_ENGINE,
        llm_provider=LLM_START_PROVIDER,
        llm_model=LLM_START_MODEL,
        no_think=NO_THINK,
        orpheus_model=TTS_ORPHEUS_MODEL,
    )

    app.state.Upsampler = UpsampleOverlap()
    _up = app.state.Upsampler
    if not _up.loudness_enabled:
        logger.info("🖥️⚙️ TTS loudness: disabled (plain resampled audio, no gain/limiting)")
    elif _up.comp_enabled:
        logger.info(f"🖥️⚙️ TTS loudness: +{_up.gain_db:.1f} dB gain, comp {_up.comp_ratio:.1f}:1 +{_up.comp_makeup_db:.1f} dB makeup, ceiling at -1 dBFS (tune via TTS_GAIN_DB / TTS_COMP_* / TTS_CEILING_DB)")
    else:
        logger.info(f"🖥️⚙️ TTS loudness: +{_up.gain_db:.1f} dB gain, comp bypassed (plain gain + limiter), ceiling at -1 dBFS (tune via TTS_GAIN_DB / TTS_COMP_ENABLED / TTS_CEILING_DB)")
    app.state.AudioInputProcessor = AudioInputProcessor(
        LANGUAGE,
        is_orpheus=TTS_START_ENGINE=="orpheus",
        pipeline_latency=app.state.SpeechPipelineManager.full_output_pipeline_latency / 1000, # seconds
    )
    app.state.Aborting = False # Keep this? Its usage isn't clear in the provided snippet. Minimizing changes.

    yield

    logger.info("🖥️⏹️ Server shutting down")
    app.state.AudioInputProcessor.shutdown()

# --------------------------------------------------------------------
# FastAPI app instance
# --------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files with no cache
app.mount("/static", NoCacheStaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.get("/favicon.ico")
async def favicon():
    """
    Serves the favicon.ico file.

    Returns:
        A FileResponse containing the favicon.
    """
    return FileResponse(str(BASE_DIR / "static" / "favicon.ico"))

@app.get("/")
async def get_index() -> HTMLResponse:
    """
    Serves the main index.html page.

    Reads the content of static/index.html and returns it as an HTML response.

    Returns:
        An HTMLResponse containing the content of index.html.
    """
    with open(BASE_DIR / "static" / "index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health() -> dict:
    """Reports TTS/LLM config plus external LLM backend reachability and models."""
    backend_ok = False
    models: list = []
    error: Optional[str] = None
    if LLM_START_PROVIDER in ("ollama", "llamacpp", "lmstudio", "openai"):
        try:
            from check_backend import list_models
            base = {
                "ollama": OLLAMA_BASE_URL,
                "llamacpp": LLAMA_BASE_URL,
            }.get(LLM_START_PROVIDER)
            models = await asyncio.to_thread(
                list_models, LLM_START_PROVIDER, base, 5.0
            )
            backend_ok = True
        except Exception as e:
            error = str(e)
    else:
        backend_ok = True  # unknown provider: nothing external to check
    # Transcription loop state: surfaced so "alive but deaf" can't hide.
    transcription_ok: Optional[bool] = None
    stt_watchdog: Optional[dict] = None
    try:
        processor = app.state.AudioInputProcessor
        transcription_ok = not processor._transcription_failed
        stt_watchdog = processor.stt_status()
    except Exception:
        pass
    return {
        "tts_engine": TTS_START_ENGINE,
        "llm_provider": LLM_START_PROVIDER,
        "llm_model": LLM_START_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "llama_base_url": LLAMA_BASE_URL,
        "backend_ok": backend_ok,
        "backend_models": models,
        # Deprecated aliases (kept for existing clients/scripts):
        "ollama_ok": backend_ok,
        "ollama_models": models,
        "transcription_ok": transcription_ok,
        "stt_watchdog": stt_watchdog,
        "error": error,
    }

# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------
def parse_json_message(text: str) -> dict:
    """
    Safely parses a JSON string into a dictionary.

    Logs a warning if the JSON is invalid and returns an empty dictionary.
    Also guards against valid JSON that is not an object (e.g. "null",
    numbers, lists): those parse fine but would crash `.get()` callers.

    Args:
        text: The JSON string to parse.

    Returns:
        A dictionary representing the parsed JSON, or an empty dictionary on error.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("🖥️⚠️ Ignoring client message with invalid JSON")
        return {}
    if not isinstance(data, dict):
        logger.warning(f"🖥️⚠️ Ignoring non-object client message: {str(text)[:80]}")
        return {}
    return data

def format_timestamp_ns(timestamp_ns: int) -> str:
    """
    Formats a nanosecond timestamp into a human-readable HH:MM:SS.fff string.

    Args:
        timestamp_ns: The timestamp in nanoseconds since the epoch.

    Returns:
        A string formatted as hours:minutes:seconds.milliseconds.
    """
    # Split into whole seconds and the nanosecond remainder
    seconds = timestamp_ns // 1_000_000_000
    remainder_ns = timestamp_ns % 1_000_000_000

    # Convert seconds part into a datetime object (local time)
    dt = datetime.fromtimestamp(seconds)

    # Format the main time as HH:MM:SS
    time_str = dt.strftime("%H:%M:%S")

    # For instance, if you want milliseconds, divide the remainder by 1e6 and format as 3-digit
    milliseconds = remainder_ns // 1_000_000
    formatted_timestamp = f"{time_str}.{milliseconds:03d}"

    return formatted_timestamp

# --------------------------------------------------------------------
# WebSocket data processing
# --------------------------------------------------------------------

async def process_incoming_data(ws: WebSocket, app: FastAPI, incoming_chunks: asyncio.Queue, callbacks: 'TranscriptionCallbacks') -> None:
    """
    Receives messages via WebSocket, processes audio and text messages.

    Handles binary audio chunks, extracting metadata (timestamp, flags) and
    putting the audio PCM data with metadata into the `incoming_chunks` queue.
    Applies back-pressure if the queue is full.
    Parses text messages (assumed JSON) and triggers actions based on message type
    (e.g., updates client TTS state via `callbacks`, clears history, sets speed).

    Args:
        ws: The WebSocket connection instance.
        app: The FastAPI application instance (for accessing global state if needed).
        incoming_chunks: An asyncio queue to put processed audio metadata dictionaries into.
        callbacks: The TranscriptionCallbacks instance for this connection to manage state.
    """
    try:
        audio_chunk_count = 0
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]

                # Ensure we have at least an 8‑byte header: 4 bytes timestamp_ms + 4 bytes flags
                if len(raw) < 8:
                    logger.warning("🖥️⚠️ Received packet too short for 8‑byte header.")
                    continue

                # Unpack big‑endian uint32 timestamp (ms) and uint32 flags
                timestamp_ms, flags = struct.unpack("!II", raw[:8])
                client_sent_ns = timestamp_ms * 1_000_000

                # Build metadata using fixed fields
                metadata = {
                    "client_sent_ms":           timestamp_ms,
                    "client_sent":              client_sent_ns,
                    "client_sent_formatted":    format_timestamp_ns(client_sent_ns),
                    "isTTSPlaying":             bool(flags & 1),
                    # Capture rate signaled by the client (defaults to the
                    # historical 48 kHz assumption for older clients).
                    "sample_rate":              callbacks.client_sample_rate,
                }

                # Record server receive time
                server_ns = time.time_ns()
                metadata["server_received"] = server_ns
                metadata["server_received_formatted"] = format_timestamp_ns(server_ns)

                # The rest of the payload is raw PCM bytes
                metadata["pcm"] = raw[8:]

                # Check queue size before putting data
                current_qsize = incoming_chunks.qsize()
                if current_qsize < MAX_AUDIO_QUEUE_SIZE:
                    # Now put only the metadata dict (containing PCM audio) into the processing queue.
                    await incoming_chunks.put(metadata)
                    audio_chunk_count += 1
                    if audio_chunk_count == 1 or audio_chunk_count % 500 == 0:
                        logger.info(f"🖥️🎙️ Audio chunks received this session: {audio_chunk_count} (latest {len(raw) - 8}B PCM @ {callbacks.client_sample_rate}Hz)")
                else:
                    # Queue is full, drop the chunk and log a warning
                    logger.warning(
                        f"🖥️⚠️ Audio queue full ({current_qsize}/{MAX_AUDIO_QUEUE_SIZE}); dropping chunk. Possible lag."
                    )

            elif "text" in msg and msg["text"]:
                # Text-based message: parse JSON
                data = parse_json_message(msg["text"])
                if not isinstance(data, dict):
                    continue
                msg_type = data.get("type")
                logger.info(Colors.apply(f"🖥️📥 ←←Client: {data}").orange)


                if msg_type == "tts_start":
                    logger.info("🖥️ℹ️ Received tts_start from client.")
                    # Update connection-specific state via callbacks
                    callbacks.tts_client_playing = True
                elif msg_type == "tts_stop":
                    logger.info("🖥️ℹ️ Received tts_stop from client.")
                    # Update connection-specific state via callbacks
                    callbacks.tts_client_playing = False
                    # Start the tail-echo hangover from the client's stop.
                    callbacks.tts_client_stop_time = time.time()
                    # Ear truth from the worklet: starved samples this answer.
                    try:
                        underruns = int(data.get("underruns", 0) or 0)
                    except (TypeError, ValueError):
                        underruns = 0
                    if underruns > 0:
                        ctx_rate = data.get("context_rate", "?")
                        try:
                            gap_ms = underruns / float(ctx_rate) * 1000.0
                            gap_str = f"~{gap_ms:.0f} ms of gaps"
                        except (TypeError, ValueError):
                            gap_str = "unknown gap duration"
                        logger.warning(
                            f"🖥️⚠️ Client starved {underruns} samples "
                            f"({gap_str}) during playback "
                            f"(context {ctx_rate} Hz). Delivery or synthesis stalled."
                        )
                # Add to the handleJSONMessage function in server.py
                elif msg_type == "clear_history":
                    logger.info("🖥️ℹ️ Received clear_history from client.")
                    app.state.SpeechPipelineManager.reset()
                elif msg_type == "set_speed":
                    speed_value = data.get("speed", 0)
                    speed_factor = speed_value / 100.0  # Convert 0-100 to 0.0-1.0
                    turn_detection = app.state.AudioInputProcessor.transcriber.turn_detection
                    if turn_detection:
                        turn_detection.update_settings(speed_factor)
                        logger.info(f"🖥️⚙️ Updated turn detection settings to factor: {speed_factor:.2f}")
                elif msg_type == "sample_rate":
                    # Client reports its actual capture rate (browsers, notably
                    # iOS, ignore the requested rate). Used for resampling.
                    try:
                        rate = int(data.get("sample_rate", 0))
                    except (TypeError, ValueError):
                        rate = 0
                    if 8000 <= rate <= 96000:
                        callbacks.client_sample_rate = rate
                        logger.info(f"🖥️⚙️ Client capture sample rate: {rate} Hz")
                    else:
                        logger.warning(f"🖥️⚠️ Ignoring invalid client sample rate: {data.get('sample_rate')}")


    except asyncio.CancelledError:
        pass # Task cancellation is expected on disconnect
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️⚠️ {Colors.apply('WARNING').red} disconnect in process_incoming_data: {repr(e)}")
    except RuntimeError as e:  # Often raised on closed transports
        logger.error(f"🖥️💥 {Colors.apply('RUNTIME_ERROR').red} in process_incoming_data: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️💥 {Colors.apply('EXCEPTION').red} in process_incoming_data: {repr(e)}")

async def send_text_messages(ws: WebSocket, message_queue: asyncio.Queue) -> None:
    """
    Continuously sends text messages from a queue to the client via WebSocket.

    Waits for messages on the `message_queue`, formats them as JSON, and sends
    them to the connected WebSocket client. Logs non-TTS messages.

    Args:
        ws: The WebSocket connection instance.
        message_queue: An asyncio queue yielding dictionaries to be sent as JSON.
    """
    try:
        while True:
            await asyncio.sleep(0.001) # Yield control
            data = await message_queue.get()
            msg_type = data.get("type")
            if msg_type != "tts_chunk":
                logger.info(Colors.apply(f"🖥️📤 →→Client: {data}").orange)
            await ws.send_json(data)
    except asyncio.CancelledError:
        pass # Task cancellation is expected on disconnect
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️⚠️ {Colors.apply('WARNING').red} disconnect in send_text_messages: {repr(e)}")
    except RuntimeError as e:  # Often raised on closed transports
        logger.error(f"🖥️💥 {Colors.apply('RUNTIME_ERROR').red} in send_text_messages: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️💥 {Colors.apply('EXCEPTION').red} in send_text_messages: {repr(e)}")

async def _reset_interrupt_flag_async(app: FastAPI, callbacks: 'TranscriptionCallbacks'):
    """
    Resets the microphone interruption flag after a delay (async version).

    Waits for 1 second, then checks if the AudioInputProcessor is still marked
    as interrupted. If so, resets the flag on both the processor and the
    connection-specific callbacks instance.

    Args:
        app: The FastAPI application instance (to access AudioInputProcessor).
        callbacks: The TranscriptionCallbacks instance for the connection.
    """
    await asyncio.sleep(1)
    # Check the AudioInputProcessor's own interrupted state
    if app.state.AudioInputProcessor.interrupted:
        logger.info(f"{Colors.apply('🖥️🎙️ ▶️ Microphone continued (async reset)').cyan}")
        app.state.AudioInputProcessor.interrupted = False
        # Reset connection-specific interruption time via callbacks
        callbacks.interruption_time = 0
        logger.info(Colors.apply("🖥️🎙️ interruption flag reset after TTS chunk (async)").cyan)

async def send_tts_chunks(app: FastAPI, message_queue: asyncio.Queue, callbacks: 'TranscriptionCallbacks') -> None:
    """
    Continuously sends TTS audio chunks from the SpeechPipelineManager to the client.

    Monitors the state of the current speech generation (if any) and the client
    connection (via `callbacks`). Retrieves audio chunks from the active generation's
    queue, upsamples/encodes them, and puts them onto the outgoing `message_queue`
    for the client. Handles the end-of-generation logic and state resets.

    Args:
        app: The FastAPI application instance (to access global components).
        message_queue: An asyncio queue to put outgoing TTS chunk messages onto.
        callbacks: The TranscriptionCallbacks instance managing this connection's state.
    """
    try:
        logger.info("🖥️🔊 Starting TTS chunk sender")
        last_quick_answer_chunk = 0
        last_chunk_sent = 0
        prev_status = None
        # Opt-in audio dump: accumulate TTS chunks per generation (no-op if off).
        dumper = get_dumper()
        tracked_gen = None
        tracked_chunks: list = []
        # Pacing clock: next wall-time a chunk may ship. Reset per
        # generation; first chunk always ships immediately (TTFA).
        pace_next_time = 0.0
        # Starvation metric: how late synthesis was vs the pace clock.
        # Reset per generation, reported at tts_end. Large values mean the
        # server itself produced audio unevenly (throughput stalls).
        pace_max_late = 0.0
        pace_chunks = 0

        while True:
            await asyncio.sleep(0.001) # Yield control

            # Stuck-playback watchdog: tts_end long ago, client still claims
            # to play, nothing flowing — flag is stale or client is wedged.
            if (callbacks.tts_end_sent_time
                    and callbacks.tts_client_playing
                    and not callbacks.tts_to_client
                    and time.time() - callbacks.tts_end_sent_time > TTS_STUCK_WARN_S
                    and time.time() - callbacks._stuck_warned_time > TTS_STUCK_WARN_S):
                callbacks._stuck_warned_time = time.time()
                logger.warning(
                    f"🖥️⚠️ Playback stuck? tts_end sent "
                    f"{time.time() - callbacks.tts_end_sent_time:.0f}s ago, "
                    f"client still reports playing, no audio flowing."
                )

            # Use connection-specific interruption_time via callbacks
            if app.state.AudioInputProcessor.interrupted and callbacks.interruption_time and time.time() - callbacks.interruption_time > 2.0:
                app.state.AudioInputProcessor.interrupted = False
                callbacks.interruption_time = 0 # Reset via callbacks
                logger.info(Colors.apply("🖥️🎙️ interruption flag reset after 2 seconds").cyan)

            # Stale-mute watchdog: interrupted with no live generation and no
            # TTS flowing means a previous session died mid-turn (eviction).
            # The per-session timer above can't fire (interruption_time=0 on
            # fresh callbacks), so clear globally instead of muting forever.
            if (app.state.AudioInputProcessor.interrupted
                    and not callbacks.interruption_time
                    and not callbacks.tts_to_client
                    and not app.state.SpeechPipelineManager.is_valid_gen()):
                app.state.AudioInputProcessor.interrupted = False
                logger.info(Colors.apply("🖥️🎙️ interruption flag reset (stale, no live generation)").cyan)

            is_tts_finished = app.state.SpeechPipelineManager.is_valid_gen() and app.state.SpeechPipelineManager.running_generation.audio_quick_finished

            # Generation boundary: save the previous generation's TTS audio.
            gen = app.state.SpeechPipelineManager.running_generation
            if gen is not tracked_gen:
                if tracked_gen is not None and tracked_chunks and dumper is not None:
                    dumper.save_assistant_turn(dumper.current_turn or 0, tracked_chunks)
                tracked_gen = gen
                tracked_chunks = []
                pace_next_time = 0.0 # New generation: first chunk ships immediately
                pace_max_late = 0.0
                pace_chunks = 0

            def log_status():
                nonlocal prev_status
                last_quick_answer_chunk_decayed = (
                    last_quick_answer_chunk
                    and time.time() - last_quick_answer_chunk > TTS_FINAL_TIMEOUT
                    and time.time() - last_chunk_sent > TTS_FINAL_TIMEOUT
                )

                curr_status = (
                    # Access connection-specific state via callbacks
                    int(callbacks.tts_to_client),
                    int(callbacks.tts_client_playing),
                    int(callbacks.tts_chunk_sent),
                    1, # Placeholder?
                    int(callbacks.is_hot), # from callbacks
                    int(callbacks.synthesis_started), # from callbacks
                    int(app.state.SpeechPipelineManager.running_generation is not None), # Global manager state
                    int(app.state.SpeechPipelineManager.is_valid_gen()), # Global manager state
                    int(is_tts_finished), # Calculated local variable
                    int(app.state.AudioInputProcessor.interrupted) # Input processor state
                )

                if curr_status != prev_status:
                    status = Colors.apply("🖥️🚦 State ").red
                    logger.info(
                        f"{status} ToClient {curr_status[0]}, "
                        f"ttsClientON {curr_status[1]}, " # Renamed slightly for clarity
                        f"ChunkSent {curr_status[2]}, "
                        f"hot {curr_status[4]}, synth {curr_status[5]}"
                        f" gen {curr_status[6]}"
                        f" valid {curr_status[7]}"
                        f" tts_q_fin {curr_status[8]}"
                        f" mic_inter {curr_status[9]}"
                    )
                    prev_status = curr_status

            # Use connection-specific state via callbacks
            if not callbacks.tts_to_client:
                await asyncio.sleep(0.001)
                log_status()
                continue

            if not app.state.SpeechPipelineManager.running_generation:
                await asyncio.sleep(0.001)
                log_status()
                continue

            if app.state.SpeechPipelineManager.running_generation.abortion_started:
                await asyncio.sleep(0.001)
                log_status()
                continue

            if not app.state.SpeechPipelineManager.running_generation.audio_quick_finished:
                app.state.SpeechPipelineManager.running_generation.tts_quick_allowed_event.set()

            if not app.state.SpeechPipelineManager.running_generation.quick_answer_first_chunk_ready:
                await asyncio.sleep(0.001)
                log_status()
                continue

            chunk = None
            try:
                chunk = app.state.SpeechPipelineManager.running_generation.audio_chunks.get_nowait()
                if chunk:
                    last_quick_answer_chunk = time.time()
            except Empty:
                final_expected = app.state.SpeechPipelineManager.running_generation.quick_answer_provided
                audio_final_finished = app.state.SpeechPipelineManager.running_generation.audio_final_finished

                if not final_expected or audio_final_finished:
                    logger.info("🖥️🏁 Sending of TTS chunks and 'user request/assistant answer' cycle finished.")
                    if pace_chunks > 0:
                        logger.info(
                            f"🖥️⏱️ Pace report: {pace_chunks} chunks, "
                            f"synthesis max {pace_max_late:.2f}s behind realtime "
                            f"({'SMOOTH' if pace_max_late < 0.15 else 'STALLY — ear hears gaps'})."
                        )
                    callbacks.send_final_assistant_answer() # Callbacks method
                    # Explicit end-of-stream: lets the client distinguish a
                    # finished generation (drain, then stop) from network
                    # starvation (keep buffering, never stop). Aborts use
                    # tts_interruption instead and need no end marker.
                    message_queue.put_nowait({
                        "type": "tts_end",
                        "content": ""
                    })
                    callbacks.tts_end_sent_time = time.time()
                    callbacks._stuck_warned_time = 0.0

                    assistant_answer = app.state.SpeechPipelineManager.running_generation.quick_answer + app.state.SpeechPipelineManager.running_generation.final_answer                    
                    app.state.SpeechPipelineManager.running_generation = None

                    callbacks.tts_chunk_sent = False # Reset via callbacks
                    callbacks.reset_state() # Reset connection state via callbacks

                await asyncio.sleep(0.001)
                log_status()
                continue

            base64_chunk = app.state.Upsampler.get_base64_chunk(chunk)
            # Realtime pacing: meter chunks against their audio duration so a
            # fast synthesizer can't pile a 30 s answer into the client's
            # buffer (tails decouple playback from generations, breaking
            # barge-in). Slow synthesis is never hurried: overdue chunks ship
            # immediately. Abort-aware: never sleep through an abort.
            chunk_seconds = len(chunk) / 2.0 / 24000.0
            if TTS_PACE_RT > 0 and chunk_seconds > 0:
                now = time.time()
                if pace_next_time <= 0:
                    pace_next_time = now + chunk_seconds * TTS_PACE_RT
                    pace_chunks += 1
                else:
                    # Starvation sample: chunk ready past its playout slot.
                    late = now - pace_next_time
                    if late > pace_max_late:
                        pace_max_late = late
                    pace_chunks += 1
                    delay = pace_next_time - now
                    pace_next_time += chunk_seconds * TTS_PACE_RT
                    while delay > 0:
                        if (app.state.SpeechPipelineManager.running_generation is not gen
                                or gen.abortion_started
                                or not callbacks.tts_to_client):
                            chunk = None
                            break
                        step = min(delay, 0.05)
                        await asyncio.sleep(step)
                        delay -= step
                if chunk is None:
                    await asyncio.sleep(0.001)
                    log_status()
                    continue
            message_queue.put_nowait({
                "type": "tts_chunk",
                "content": base64_chunk,
                # Wire contract: playback must resample from this rate to
                # the device rate (see ttsPlaybackProcessor.js). Keeps
                # 44.1 kHz BT speakers / 16 kHz HFP routes in tune.
                "sample_rate": app.state.Upsampler.TARGET_RATE,
            })
            last_chunk_sent = time.time()
            if dumper is not None and tracked_gen is not None:
                tracked_chunks.append(chunk)

            # Use connection-specific state via callbacks
            if not callbacks.tts_chunk_sent:
                # Use the async helper function instead of a thread
                asyncio.create_task(_reset_interrupt_flag_async(app, callbacks))

            callbacks.tts_chunk_sent = True # Set via callbacks

    except asyncio.CancelledError:
        pass # Task cancellation is expected on disconnect
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️⚠️ {Colors.apply('WARNING').red} disconnect in send_tts_chunks: {repr(e)}")
    except RuntimeError as e:
        logger.error(f"🖥️💥 {Colors.apply('RUNTIME_ERROR').red} in send_tts_chunks: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️💥 {Colors.apply('EXCEPTION').red} in send_tts_chunks: {repr(e)}")


# --------------------------------------------------------------------
# Callback class to handle transcription events
# --------------------------------------------------------------------
class TranscriptionCallbacks:
    """
    Manages state and callbacks for a single WebSocket connection's transcription lifecycle.

    This class holds connection-specific state flags (like TTS status, user interruption)
    and implements callback methods triggered by the `AudioInputProcessor` and
    `SpeechPipelineManager`. It sends messages back to the client via the provided
    `message_queue` and manages interaction logic like interruptions and final answer delivery.
    It also includes a threaded worker to handle abort checks based on partial transcription.
    """
    def __init__(self, app: FastAPI, message_queue: asyncio.Queue):
        """
        Initializes the TranscriptionCallbacks instance for a WebSocket connection.

        Args:
            app: The FastAPI application instance (to access global components).
            message_queue: An asyncio queue for sending messages back to the client.
        """
        self.app = app
        self.message_queue = message_queue
        self.final_transcription = ""
        self.abort_text = ""
        self.last_abort_text = ""

        # Initialize connection-specific state flags here
        self.tts_to_client: bool = False
        self.user_interrupted: bool = False
        self.tts_chunk_sent: bool = False
        self.tts_client_playing: bool = False
        self.interruption_time: float = 0.0
        # Last time the client reported TTS stop (tail-echo hangover basis).
        self.tts_client_stop_time: float = 0.0
        # Stuck-playback watchdog state (see send_tts_chunks).
        self.tts_end_sent_time: float = 0.0
        self._stuck_warned_time: float = 0.0
        # Barge-in arming: VAD fired while TTS was audible; abort waits for
        # text confirmation (echo guard) instead of firing on the blip.
        self._barge_armed: bool = False
        self._echo_suppress_count: int = 0
        # Fusion-gate clock: episode start for sustained-audio evidence.
        # 0.0 = no episode.
        self._barge_arm_time: float = 0.0
        # Recent assistant output for echo matching: deque of (timestamp, text).
        # Time-pruned, so it survives generation resets without going stale.
        self.recent_assistant_segments: deque = deque(maxlen=100)
        # Re-emission suppression: VAD-episode start for speech-rate check.
        # 0.0 = no episode. Set on first VAD, consumed at turn end.
        self._episode_start_time: float = 0.0
        # Set when a turn is dropped as re-emission so on_final doesn't store it.
        self._turn_dropped: bool = False
        # Capture rate signaled by the client via "sample_rate" message.
        # Defaults to the historical 48 kHz assumption for older clients.
        # NOTE: reset_state() intentionally does not touch this.
        self.client_sample_rate: int = 48000

        # These were already effectively instance variables or reset logic existed
        self.silence_active: bool = True
        self.is_hot: bool = False
        self.user_finished_turn: bool = False
        self.synthesis_started: bool = False
        self.assistant_answer: str = ""
        self.final_assistant_answer: str = ""
        self.is_processing_potential: bool = False
        self.is_processing_final: bool = False
        self.last_inferred_transcription: str = ""
        self.final_assistant_answer_sent: bool = False
        self.partial_transcription: str = "" # Added for clarity

        self.reset_state() # Call reset to ensure consistency

        self.abort_request_event = threading.Event()
        self.abort_worker_thread = threading.Thread(target=self._abort_worker, name="AbortWorker", daemon=True)
        self.abort_worker_thread.start()


    def reset_state(self):
        """Resets connection-specific state flags and variables to their initial values."""
        # Reset all connection-specific state flags
        self.tts_to_client = False
        self.user_interrupted = False
        self.tts_chunk_sent = False
        # Don't reset tts_client_playing here, it reflects client state reports
        self.interruption_time = 0.0

        # Reset other state variables
        self.silence_active = True
        self.is_hot = False
        self.user_finished_turn = False
        self.synthesis_started = False
        self.assistant_answer = ""
        self.final_assistant_answer = ""
        self.is_processing_potential = False
        self.is_processing_final = False
        self.last_inferred_transcription = ""
        self.final_assistant_answer_sent = False
        self.partial_transcription = ""
        # A finished generation means any armed barge-in is over (its TTS is
        # done; further speech is a fresh turn, judged on its own words).
        self._barge_armed = False
        self._echo_suppress_count = 0
        self._barge_arm_time = 0.0
        self._episode_start_time = 0.0
        self._turn_dropped = False

        # Keep the abort call related to the audio processor/pipeline manager
        self.app.state.AudioInputProcessor.abort_generation()


    # ------------------------------------------------------------------
    # Echo (self-interrupt) guard
    # ------------------------------------------------------------------
    # STT emits contractions ("you're") while TTS source text usually has
    # them expanded ("you are"). Expand before comparing, or echo fragments
    # systematically fail to match.
    _CONTRACTIONS = {
        "you're": "you are", "we're": "we are", "they're": "they are",
        "i'm": "i am", "i'll": "i will", "i've": "i have", "i'd": "i would",
        "it's": "it is", "that's": "that is", "what's": "what is",
        "let's": "let us", "don't": "do not", "doesn't": "does not",
        "didn't": "did not", "can't": "can not", "won't": "will not",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "haven't": "have not", "hasn't": "has not",
        "hadn't": "had not", "couldn't": "could not", "wouldn't": "would not",
        "shouldn't": "should not", "there's": "there is",
    }

    @classmethod
    def _norm_words(cls, txt: str) -> list:
        """Lowercase alphanumeric word tokens for echo comparison."""
        import re as _re
        toks = _re.findall(r"[a-z0-9']+", (txt or "").lower())
        out: list = []
        for t in toks:
            exp = cls._CONTRACTIONS.get(t)
            out.extend(exp.split() if exp else [t.strip("'")])
        return [w for w in out if w]

    def _note_assistant_text(self, txt: str) -> None:
        """Records assistant output (partial or final) for echo matching."""
        if not txt or not txt.strip():
            return
        now = time.time()
        self.recent_assistant_segments.append((now, txt))
        # Prune anything too old to be in the air anymore.
        cutoff = now - (TTS_ECHO_WINDOW_S + 2.0)
        while self.recent_assistant_segments and self.recent_assistant_segments[0][0] < cutoff:
            self.recent_assistant_segments.popleft()

    def _echo_window_words(self) -> list:
        """Assistant words produced long enough ago to have played, recently
        enough to still be echoing (acoustic flight + buffers + tail)."""
        now = time.time()
        words: list = []
        for ts, txt in self.recent_assistant_segments:
            age = now - ts
            if TTS_ECHO_MIN_DELAY_S <= age <= TTS_ECHO_WINDOW_S:
                words.extend(self._norm_words(txt))
        return words

    def is_tts_audible(self) -> bool:
        """True if the client is (probably) audibly playing TTS right now:
        it says so, or it stopped only moments ago (room/speaker tail)."""
        if self.tts_client_playing:
            return True
        return (time.time() - self.tts_client_stop_time) < TTS_ECHO_HANGOVER_S

    def _is_likely_echo(self, txt: str) -> tuple:
        """Decides whether a mic partial is the assistant's own echo.

        Echo usually transcribes as fragments of just-played assistant words,
        so a word-ordered subsequence match (or strong fuzzy match for longer
        partials) against the recent assistant window means echo. Genuine
        barge-ins use different words and pass through with zero added delay.

        Returns (is_echo, reason).
        """
        if not TTS_ECHO_ENABLED:
            return False, "guard-disabled"
        words = self._norm_words(txt)
        if not words:
            return False, "empty-partial"
        window = self._echo_window_words()
        if not window:
            return False, "no-assistant-window"
        # Ordered subsequence: every partial word appears in window order.
        wi = 0
        matched = 0
        for w in words:
            try:
                wi = window.index(w, wi) + 1
                matched += 1
            except ValueError:
                break
        if matched == len(words):
            if len(words) <= 2:
                return True, f"short-fragment-match ({' '.join(words)})"
            return True, f"subsequence-match ({len(words)} words)"
        if len(words) >= 3:
            ratio = SequenceMatcher(None, " ".join(words), " ".join(window)).ratio()
            if ratio >= 0.6:
                return True, f"fuzzy-match ({ratio:.2f})"
        return False, "words-diverge"

    def _barge_audit(self, trigger: str, txt: str = "", verdict: str = "") -> None:
        """One-line audit for every barge-in decision: what fired, on what
        evidence, and what TTS state the server saw. This is how echo vs
        genuine interruptions are told apart after the fact."""
        try:
            mgr = self.app.state.SpeechPipelineManager
            gen = mgr.running_generation
            gen_state = f"gen={int(gen is not None)} valid={int(mgr.is_valid_gen())}"
        except Exception:
            gen_state = "gen=? valid=?"
        logger.info(
            f"🖥️🔍 BARGEAUDIT trigger={trigger} verdict={verdict} "
            f"ttsClient={int(self.tts_client_playing)} ttsToClient={int(self.tts_to_client)} "
            f"armed={int(self._barge_armed)} {gen_state} partial='{txt[:80]}'"
        )

    def _emit_barge_signals(self, reason: str) -> None:
        """Client-side half of a confirmed barge-in: stop server streaming,
        mark interruption, finalize any partial answer, and tell the client
        to clear its playback buffer. The generation abort itself stays with
        check_abort (similarity-protected). Safe to call when nothing is
        playing (messages just clear an empty buffer)."""
        self.tts_to_client = False
        self.user_interrupted = True
        logger.info(Colors.apply("🖥️✅ Sending final assistant answer (forced on interruption)").pink)
        self.send_final_assistant_answer(forced=True)
        self.tts_chunk_sent = False
        logger.info("🖥️🛑 Sending stop_tts to client.")
        self.message_queue.put_nowait({"type": "stop_tts", "content": ""})
        logger.info("🖥️❗ Sending tts_interruption to client.")
        self.message_queue.put_nowait({"type": "tts_interruption", "content": ""})
        # The client is about to stop: start the tail hangover now.
        self.tts_client_stop_time = time.time()
        logger.info(f"{Colors.apply('🖥️🛑 BARGE-IN confirmed:').red} {reason}")

    def _barge_voiced_ms(self) -> float:
        """Sustained speech-like audio (ms) fed since arming, for the fusion
        gate. Fail-open: on any lookup error returns a large value so
        barge-in still works on words + echo-match alone (the audio leg is
        enhancement, not load-bearing)."""
        try:
            return self.app.state.AudioInputProcessor.voiced_ms_since(self._barge_arm_time or 0.0)
        except Exception as e:
            logger.debug(f"🖥️⚠️ voiced-ms lookup failed, bypassing audio leg: {e}")
            return 1e9

    def _abort_worker(self):
        """Background thread worker to check for abort conditions based on partial text."""
        while True:
            was_set = self.abort_request_event.wait(timeout=0.1) # Check every 100ms
            if was_set:
                self.abort_request_event.clear()
                # Only trigger abort check if the text actually changed
                if self.last_abort_text != self.abort_text:
                    self.last_abort_text = self.abort_text
                    txt = self.abort_text
                    if self._barge_armed:
                        # Fusion gate: sustained speech-like audio AND
                        # divergent words. A blip that never becomes words
                        # stays armed and waits; words without sustained
                        # audio (hallucinated from a thump) wait too.
                        words = self._norm_words(txt)
                        if not words:
                            self._barge_audit("partial", txt, "SKIPPED empty (awaiting words)")
                            continue
                        echo, why = self._is_likely_echo(txt)
                        if echo and self._echo_suppress_count < TTS_ECHO_MAX_SUPPRESS:
                            self._echo_suppress_count += 1
                            self._barge_audit("partial", txt, f"SUPPRESSED echo ({why}) #{self._echo_suppress_count}")
                            continue
                        voiced = self._barge_voiced_ms()
                        if voiced < TTS_BARGE_MIN_VOICED_MS:
                            self._barge_audit("partial", txt, f"WAITING audio (voiced={voiced:.0f}ms words={len(words)})")
                            continue
                        # Genuine: sustained speech with divergent words.
                        self._barge_armed = False
                        self._echo_suppress_count = 0
                        self._barge_arm_time = 0.0
                        self._barge_audit("partial", txt, f"GENUINE barge-in (voiced={voiced:.0f}ms words={len(words)})")
                        self.send_final_assistant_answer(forced=True)
                    else:
                        logger.debug(f"🖥️🧠 Abort check triggered by partial: '{txt}'")
                    aborted = self.app.state.SpeechPipelineManager.check_abort(txt, False, "on_partial")
                    # VAD-before-TTS race: abort decided against playing audio
                    # this check didn't know about — silence the client too.
                    if aborted and (self.tts_client_playing or self.tts_to_client):
                        self._emit_barge_signals("late VAD race, text-confirmed")

    def on_partial(self, txt: str):
        """
        Callback invoked when a partial transcription result is available.

        Updates internal state, sends the partial result to the client,
        and signals the abort worker thread to check for potential interruptions.

        Args:
            txt: The partial transcription text.
        """
        self.final_assistant_answer_sent = False # New user speech invalidates previous final answer sending state
        self.final_transcription = "" # Clear final transcription as this is partial
        self.partial_transcription = txt
        self.message_queue.put_nowait({"type": "partial_user_request", "content": txt})
        self.abort_text = txt # Update text used for abort check
        # Word floor: an empty partial (noise blip) must not even reach the
        # abort worker — similarity against "" reads as divergent and would
        # kill a live generation on zero spoken words.
        if self._norm_words(txt):
            self.abort_request_event.set() # Signal the abort worker

    def safe_abort_running_syntheses(self, reason: str):
        """Placeholder for safely aborting syntheses (currently does nothing)."""
        # TODO: Implement actual abort logic if needed, potentially interacting with SpeechPipelineManager
        pass

    def on_tts_allowed_to_synthesize(self):
        """Callback invoked when the system determines TTS synthesis can proceed."""
        # Access global manager state
        if self.app.state.SpeechPipelineManager.running_generation and not self.app.state.SpeechPipelineManager.running_generation.abortion_started:
            logger.info(f"{Colors.apply('🖥️🔊 TTS ALLOWED').blue}")
            self.app.state.SpeechPipelineManager.running_generation.tts_quick_allowed_event.set()

    def on_potential_sentence(self, txt: str):
        """
        Callback invoked when a potentially complete sentence is detected by the STT.

        Triggers the preparation of a speech generation based on this potential sentence.
        If the client is still playing stale audio, a genuine new sentence
        preempts it first (clear + drain) so answers can't pile up on top of
        each other. Echo-like sentences never preempt.

        Args:
            txt: The potential sentence text.
        """
        logger.debug(f"🖥️🧠 Potential sentence: '{txt}'")
        if TTS_PREEMPT_ENABLED and (self.tts_client_playing or self.tts_to_client):
            echo, why = self._is_likely_echo(txt)
            if echo:
                self._barge_audit("potential", txt, f"SKIPPED preempt (echo {why})")
            else:
                self._barge_audit("potential", txt, "PREEMPT stale playback")
                # Clear the client's stale buffer; new audio follows.
                # tts_interruption (not stop_tts): keeps the door open for
                # the new generation's chunks instead of muting them.
                # NOTE: tts_to_client is deliberately left alone — streaming
                # ownership stays with the generation lifecycle (abort ends
                # it). Pausing it here could permanently cut an answer whose
                # replacement prepare gets debounced.
                self.message_queue.put_nowait({"type": "tts_interruption", "content": ""})
                self.tts_client_stop_time = time.time()
                # Drain queued stale audio so the new turn starts clean.
                # Synthesis threads keep running until check_abort catches up;
                # their output lands in a queue nobody sends from anymore.
                try:
                    mgr = self.app.state.SpeechPipelineManager
                    if mgr.running_generation is not None:
                        q = mgr.running_generation.audio_chunks
                        dropped = 0
                        while True:
                            try:
                                q.get_nowait()
                                dropped += 1
                            except Empty:
                                break
                        if dropped:
                            logger.info(f"🖥️🧹 Preempt dropped {dropped} stale audio chunks.")
                except Exception as e:
                    logger.warning(f"🖥️⚠️ Preempt drain skipped: {e}")
        # Access global manager state
        self.app.state.SpeechPipelineManager.prepare_generation(txt)

    def on_potential_final(self, txt: str):
        """
        Callback invoked when a potential *final* transcription is detected (hot state).

        Logs the potential final transcription.

        Args:
            txt: The potential final transcription text.
        """
        logger.info(f"{Colors.apply('🖥️🧠 HOT: ').magenta}{txt}")

    def on_potential_abort(self):
        """Callback invoked if the STT detects a potential need to abort based on user speech."""
        # Placeholder: Currently logs nothing, could trigger abort logic.
        pass

    def on_before_final(self, audio: bytes, txt: str):
        """
        Callback invoked just before the final STT result for a user turn is confirmed.

        Sets flags indicating user finished, allows TTS if pending, interrupts microphone input,
        releases TTS stream to client, sends final user request and any pending partial
        assistant answer to the client, and adds user request to history.

        Args:
            audio: The raw audio bytes corresponding to the final transcription. (Currently unused)
            txt: The transcription text (might be slightly refined in on_final).
        """
        logger.info(Colors.apply('🖥️🏁 =================== USER TURN END ===================').light_gray)

        # Send final user request (using the reliable final_transcription OR current partial if final isn't set yet)
        user_request_content = self.final_transcription if self.final_transcription else self.partial_transcription

        # Re-emission suppression: a final implying an un-speakable rate over
        # the VAD episode is a stale STT buffer finalized twice, not fresh
        # speech (observed: 10 words from a 0.9 s episode). Drop it as a turn:
        # no message, no history, no downstream prepare effects from here.
        # Exact duplicates at plausible rates (frustrated repeats) still pass.
        _turn_words = self._norm_words(user_request_content)
        _now = time.time()
        _ep_dur = _now - self._episode_start_time if self._episode_start_time > 0 else 0.0
        self._episode_start_time = 0.0
        if _turn_words and _ep_dur > 0:
            _wps = len(_turn_words) / max(_ep_dur, 1e-6)
            if _wps > TTS_TURN_MAX_WPS:
                self._turn_dropped = True
                self._barge_audit("turn", user_request_content, f"IGNORED re-emission ({len(_turn_words)} words in {_ep_dur:.1f}s = {_wps:.1f} wps)")
                return

        self.user_finished_turn = True
        self.user_interrupted = False # Reset connection-specific flag (user finished, not interrupted)
        # Access global manager state
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            logger.info(f"{Colors.apply('🖥️🔊 TTS ALLOWED (before final)').blue}")
            self.app.state.SpeechPipelineManager.running_generation.tts_quick_allowed_event.set()

        # first block further incoming audio (Audio processor's state)
        if not self.app.state.AudioInputProcessor.interrupted:
            logger.info(f"{Colors.apply('🖥️🎙️ ⏸️ Microphone interrupted (end of turn)').cyan}")
            self.app.state.AudioInputProcessor.interrupted = True
            self.interruption_time = time.time() # Set connection-specific flag

        logger.info(f"{Colors.apply('🖥️🔊 TTS STREAM RELEASED').blue}")
        self.tts_to_client = True # Set connection-specific flag
        self.tts_end_sent_time = 0.0 # New audio flowing: previous end is history
        self._stuck_warned_time = 0.0

        # Send final user request (using the reliable final_transcription OR current partial if final isn't set yet)
        user_request_content = self.final_transcription if self.final_transcription else self.partial_transcription
        self.message_queue.put_nowait({
            "type": "final_user_request",
            "content": user_request_content
        })

        # Access global manager state
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            # Send partial assistant answer (if available) to the client
            # Use connection-specific user_interrupted flag
            if self.app.state.SpeechPipelineManager.running_generation.quick_answer and not self.user_interrupted:
                self.assistant_answer = self.app.state.SpeechPipelineManager.running_generation.quick_answer
                self.message_queue.put_nowait({
                    "type": "partial_assistant_answer",
                    "content": self.assistant_answer
                })

        logger.info(f"🖥️🧠 Adding user request to history: '{user_request_content}'")
        # Access global manager state
        self.app.state.SpeechPipelineManager.history.append({"role": "user", "content": user_request_content})

        # Opt-in audio dump: persist the exact STT input for this turn + manifest.
        dumper = get_dumper()
        if dumper:
            turn = dumper.next_turn()
            audio_file = None
            try:
                import numpy as np
                audio = self.app.state.AudioInputProcessor.transcriber.get_last_audio_copy()
                if audio is not None and len(audio) > 0:
                    pcm16 = (np.clip(np.asarray(audio), -1.0, 1.0) * 32768).astype(np.int16)
                    audio_file = dumper.save_user_turn(turn, pcm16)
            except Exception as e:
                logger.warning(f"💾 user turn audio dump skipped: {e}")
            dumper.log_event({"event": "user_turn", "turn": turn,
                              "text": user_request_content, "audio_file": audio_file,
                              "sample_rate": self.app.state.AudioInputProcessor.last_sample_rate,
                              "speaker": self._speaker_policy_trace()})

    def _speaker_policy_trace(self) -> dict:
        """Attach the latest flag-only speaker score + policy decision.

        Observability only: the decision is logged to the manifest. Behavior
        is unchanged unless SPEAKER_ENFORCE=true (see speaker_policy).
        """
        try:
            from speaker_policy import decide, should_enforce
            score = self.app.state.AudioInputProcessor.last_speaker_score
            decision = decide(score)
            decision["enforced"] = should_enforce()
            if score:
                decision["window_s"] = score.get("window_s")
            return decision
        except Exception as e:
            logger.debug(f"💾 speaker policy trace skipped: {e}")
            return {"action": "obey", "reason": "trace-error", "score": None,
                    "enforced": False}

    def on_final(self, txt: str):
        """
        Callback invoked when the final transcription result for a user turn is available.

        Logs the final transcription and stores it.

        Args:
            txt: The final transcription text.
        """
        logger.info(f"\n{Colors.apply('🖥️✅ FINAL USER REQUEST (STT Callback): ').green}{txt}")
        if self._turn_dropped:
            # Companion to a re-emission drop in on_before_final: don't store
            # what was never a turn. Barge judgment below still runs.
            self._turn_dropped = False
            self._barge_audit("final", txt, "IGNORED re-emission (turn dropped)")
        elif not self.final_transcription: # Store it if not already set by on_before_final logic
             self.final_transcription = txt
        # Short-barge safety net: VAD armed during audible TTS but no partial
        # ever confirmed it (single clipped word, then silence). Judge the
        # final text the same way rather than leaving the arm stale.
        # Word floor applies here too: an empty final confirms nothing.
        # Fusion gate applies as well — but a final is terminal, so thin
        # voice evidence disarms (IGNORED) instead of waiting.
        if self._barge_armed:
            if not self._norm_words(txt):
                self._barge_audit("final", txt, "SKIPPED empty (awaiting words)")
            else:
                echo, why = self._is_likely_echo(txt)
                if echo:
                    self._barge_armed = False
                    self._echo_suppress_count = 0
                    self._barge_arm_time = 0.0
                    self._barge_audit("final", txt, f"SUPPRESSED echo ({why})")
                    return
                voiced = self._barge_voiced_ms()
                if voiced < TTS_BARGE_MIN_VOICED_MS:
                    self._barge_armed = False
                    self._echo_suppress_count = 0
                    self._barge_arm_time = 0.0
                    self._barge_audit("final", txt, f"IGNORED thin-audio (voiced={voiced:.0f}ms)")
                    return
                self._barge_armed = False
                self._echo_suppress_count = 0
                self._barge_arm_time = 0.0
                self._barge_audit("final", txt, f"GENUINE barge-in (voiced={voiced:.0f}ms)")
                self.send_final_assistant_answer(forced=True)
                aborted = self.app.state.SpeechPipelineManager.check_abort(txt, False, "on_final(barge)")
                if aborted and (self.tts_client_playing or self.tts_to_client):
                    self._emit_barge_signals("final-text-confirmed")

    def abort_generations(self, reason: str):
        """
        Triggers the abortion of any ongoing speech generation process.

        Logs the reason and calls the SpeechPipelineManager's abort method.

        Args:
            reason: A string describing why the abortion is triggered.
        """
        logger.info(f"{Colors.apply('🖥️🛑 Aborting generation:').blue} {reason}")
        # Access global manager state
        self.app.state.SpeechPipelineManager.abort_generation(reason=f"server.py abort_generations: {reason}")

    def on_silence_active(self, silence_active: bool):
        """
        Callback invoked when the silence detection state changes.

        Updates the internal silence_active flag.

        Args:
            silence_active: True if silence is currently detected, False otherwise.
        """
        # logger.debug(f"🖥️🎙️ Silence active: {silence_active}") # Optional: Can be noisy
        self.silence_active = silence_active

    def on_partial_assistant_text(self, txt: str):
        """
        Callback invoked when a partial text result from the assistant (LLM) is available.

        Updates the internal assistant answer state and sends the partial answer to the client,
        unless the user has interrupted.

        Args:
            txt: The partial assistant text.
        """
        logger.info(f"{Colors.apply('🖥️💬 PARTIAL ASSISTANT ANSWER: ').green}{txt}")
        # Record output for the echo guard regardless of interrupt state: if
        # this text gets synthesized, the mic may hear it back.
        self._note_assistant_text(txt)
        # Use connection-specific user_interrupted flag
        if not self.user_interrupted:
            self.assistant_answer = txt
            # Use connection-specific tts_to_client flag
            if self.tts_to_client:
                self.message_queue.put_nowait({
                    "type": "partial_assistant_answer",
                    "content": txt
                })

    def on_recording_start(self):
        """
        Callback invoked when the audio input processor starts recording user speech.

        Barge-in policy (echo-safe, still instant for real speech):
        - Nothing audible (idle, or server streaming into the client's
          prebuffer where nothing can echo yet): nothing to abort; speech
          flows to STT normally. This preserves the instant head-hole case.
        - Client audibly playing with an active generation: ARM only. A bare
          VAD blip is exactly what speaker echo looks like, so the abort
          waits for the first partial's words (echo guard in _abort_worker).
        - Client audibly playing but nothing active (tail drain): pure echo,
          ignore it — and don't poison flags with an interrupt-of-nothing.
        """
        audible = self.is_tts_audible()
        try:
            active = bool(self.tts_to_client or self.app.state.SpeechPipelineManager.is_valid_gen())
        except Exception:
            active = False
        logger.info(f"{Colors.ORANGE}🖥️🎙️ Recording started.{Colors.RESET} TTS Client Playing: {self.tts_client_playing} audible={audible} serverActive={active}")
        # Episode clock for re-emission suppression: first VAD starts it.
        # (Trailing silence inflates the span, so real speech rates read low
        # — the check fails open toward accepting.)
        if self._episode_start_time <= 0:
            self._episode_start_time = time.time()
        if not audible:
            return
        if not active:
            # Tail echo after the generation already finished: nothing to
            # kill. Audit it and leave all flags alone.
            self._barge_audit("vad", "", "IGNORED tail (no active generation)")
            return
        # Audible TTS + live generation: arm, decide on words, not blips.
        # The fusion clock starts now; sustained-audio evidence accrues after.
        self._barge_armed = True
        self._barge_arm_time = time.time()
        self._barge_audit("vad", "", "ARMED (awaiting text confirmation)")

    def send_final_assistant_answer(self, forced=False):
        """
        Sends the final (or best available) assistant answer to the client.

        Constructs the full answer from quick and final parts if available.
        If `forced` and no full answer exists, uses the last partial answer.
        Cleans the text and sends it as 'final_assistant_answer' if not already sent.

        Args:
            forced: If True, attempts to send the last partial answer if no complete
                    final answer is available. Defaults to False.
        """
        final_answer = ""
        # Access global manager state
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            final_answer = self.app.state.SpeechPipelineManager.running_generation.quick_answer + self.app.state.SpeechPipelineManager.running_generation.final_answer

        if not final_answer: # Check if constructed answer is empty
            # If forced, try using the last known partial answer from this connection
            if forced and self.assistant_answer:
                 final_answer = self.assistant_answer
                 logger.warning(f"🖥️⚠️ Using partial answer as final (forced): '{final_answer}'")
            else:
                logger.warning(f"🖥️⚠️ Final assistant answer was empty, not sending.")
                return# Nothing to send

        logger.debug(f"🖥️✅ Attempting to send final answer: '{final_answer}' (Sent previously: {self.final_assistant_answer_sent})")

        if not self.final_assistant_answer_sent and final_answer:
            import re
            # Clean up the final answer text
            cleaned_answer = re.sub(r'[\r\n]+', ' ', final_answer)
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()
            cleaned_answer = cleaned_answer.replace('\\n', ' ')
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()

            if cleaned_answer: # Ensure it's not empty after cleaning
                logger.info(f"\n{Colors.apply('🖥️✅ FINAL ASSISTANT ANSWER (Sending): ').green}{cleaned_answer}")
                # Record output for the echo guard: this text is about to play.
                self._note_assistant_text(cleaned_answer)
                self.message_queue.put_nowait({
                    "type": "final_assistant_answer",
                    "content": cleaned_answer
                })
                app.state.SpeechPipelineManager.history.append({"role": "assistant", "content": cleaned_answer})
                self.final_assistant_answer_sent = True
                self.final_assistant_answer = cleaned_answer # Store the sent answer
                # Opt-in audio dump: manifest entry for the assistant turn.
                dumper = get_dumper()
                if dumper and dumper.current_turn:
                    dumper.log_event({"event": "assistant_turn", "turn": dumper.current_turn,
                                      "text": cleaned_answer,
                                      "audio_file": f"assistant_turn_{dumper.current_turn:02d}.wav"})
            else:
                logger.warning(f"🖥️⚠️ {Colors.YELLOW}Final assistant answer was empty after cleaning.{Colors.RESET}")
                self.final_assistant_answer_sent = False # Don't mark as sent
                self.final_assistant_answer = "" # Clear the stored answer
        elif forced and not final_answer: # Should not happen due to earlier check, but safety
             logger.warning(f"🖥️⚠️ {Colors.YELLOW}Forced send of final assistant answer, but it was empty.{Colors.RESET}")
             self.final_assistant_answer = "" # Clear the stored answer


# --------------------------------------------------------------------
# Exclusive session management
# --------------------------------------------------------------------
# The audio pipeline (AudioInputProcessor, transcriber callbacks,
# SpeechPipelineManager) is global and single-user: every connection
# reassigns its callbacks. To prevent two clients from silently corrupting
# each other, only one voice session is active at a time — a new connection
# evicts the previous one with a notice the UI can display.
active_ws_lock = threading.Lock()
current_client_ws: Optional[WebSocket] = None


# --------------------------------------------------------------------
# Main WebSocket endpoint
# --------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Handles the main WebSocket connection for real-time voice chat.

    Accepts a connection, sets up connection-specific state via `TranscriptionCallbacks`,
    initializes audio/message queues, and creates asyncio tasks for handling
    incoming data, audio processing, outgoing text messages, and outgoing TTS chunks.
    Manages the lifecycle of these tasks and cleans up on disconnect.

    Args:
        ws: The WebSocket connection instance provided by FastAPI.
    """
    await ws.accept()
    logger.info("🖥️✅ Client connected via WebSocket.")

    # Exclusive session: evict any previous client so its orphaned tasks
    # can't fight over the global pipeline callbacks.
    global current_client_ws
    with active_ws_lock:
        previous_ws = current_client_ws
        current_client_ws = ws
    if previous_ws is not None and previous_ws is not ws:
        logger.info("🖥️🔀 New client connected; evicting previous session.")
        try:
            await previous_ws.send_json({
                "type": "session_takeover",
                "content": "Another device took over this voice session. Press Start to reclaim it.",
            })
        except Exception:
            pass  # already gone; close below is best-effort too
        try:
            await previous_ws.close(code=4000)
        except Exception:
            pass

    message_queue = asyncio.Queue()
    audio_chunks = asyncio.Queue()

    # Fresh session must start unmuted: `interrupted` is global on
    # AudioInputProcessor but its reset lived in the previous session's
    # TTS-sender task, which eviction cancels before it fires. Without this,
    # one interrupted turn + a device switch mutes all future sessions
    # (audio_in drops chunks while interrupted).
    try:
        if app.state.AudioInputProcessor.interrupted:
            logger.info("🖥️🎙️ Clearing stale global interruption on new session.")
        app.state.AudioInputProcessor.interrupted = False
    except Exception:
        pass

    # Set up callback manager - THIS NOW HOLDS THE CONNECTION-SPECIFIC STATE
    callbacks = TranscriptionCallbacks(app, message_queue)

    # Assign callbacks to the AudioInputProcessor (global component)
    # These methods within callbacks will now operate on its *instance* state
    app.state.AudioInputProcessor.realtime_callback = callbacks.on_partial
    app.state.AudioInputProcessor.transcriber.potential_sentence_end = callbacks.on_potential_sentence
    app.state.AudioInputProcessor.transcriber.on_tts_allowed_to_synthesize = callbacks.on_tts_allowed_to_synthesize
    app.state.AudioInputProcessor.transcriber.potential_full_transcription_callback = callbacks.on_potential_final
    app.state.AudioInputProcessor.transcriber.potential_full_transcription_abort_callback = callbacks.on_potential_abort
    app.state.AudioInputProcessor.transcriber.full_transcription_callback = callbacks.on_final
    app.state.AudioInputProcessor.transcriber.before_final_sentence = callbacks.on_before_final
    app.state.AudioInputProcessor.recording_start_callback = callbacks.on_recording_start
    app.state.AudioInputProcessor.silence_active_callback = callbacks.on_silence_active

    # Assign callback to the SpeechPipelineManager (global component)
    app.state.SpeechPipelineManager.on_partial_assistant_text = callbacks.on_partial_assistant_text

    # Create tasks for handling different responsibilities
    # Pass the 'callbacks' instance to tasks that need connection-specific state
    tasks = [
        asyncio.create_task(process_incoming_data(ws, app, audio_chunks, callbacks)), # Pass callbacks
        asyncio.create_task(app.state.AudioInputProcessor.process_chunk_queue(audio_chunks)),
        asyncio.create_task(send_text_messages(ws, message_queue)),
        asyncio.create_task(send_tts_chunks(app, message_queue, callbacks)), # Pass callbacks
    ]

    try:
        # Wait for any task to complete (e.g., client disconnect)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            if not task.done():
                task.cancel()
        # Await cancelled tasks to let them clean up if needed
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception as e:
        logger.error(f"🖥️💥 {Colors.apply('ERROR').red} in WebSocket session: {repr(e)}")
    finally:
        logger.info("🖥️🧹 Cleaning up WebSocket tasks...")
        for task in tasks:
            if not task.done():
                task.cancel()
        # Ensure all tasks are awaited after cancellation
        # Use return_exceptions=True to prevent gather from stopping on first error during cleanup
        await asyncio.gather(*tasks, return_exceptions=True)
        # Release the exclusive slot if it is still ours (a newer client may
        # have already taken over; never clear someone else's session).
        with active_ws_lock:
            if current_client_ws is ws:
                current_client_ws = None
        logger.info("🖥️❌ WebSocket session ended.")

# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------
if __name__ == "__main__":

    # Run the server without SSL
    if not USE_SSL:
        logger.info("🖥️▶️ Starting server without SSL.")
        uvicorn.run("server:app", host="0.0.0.0", port=8000, log_config=None)

    else:
        logger.info("🖥️🔒 Attempting to start server with SSL.")
        # Check if cert files exist
        cert_file = "127.0.0.1+1.pem"
        key_file = "127.0.0.1+1-key.pem"
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
             logger.error(f"🖥️💥 SSL cert file ({cert_file}) or key file ({key_file}) not found.")
             logger.error("🖥️💥 Please generate them using mkcert:")
             logger.error("🖥️💥   choco install mkcert") # Assuming Windows based on earlier check, adjust if needed
             logger.error("🖥️💥   mkcert -install")
             logger.error("🖥️💥   mkcert 127.0.0.1 YOUR_LOCAL_IP") # Remind user to replace with actual IP if needed
             logger.error("🖥️💥 Exiting.")
             sys.exit(1)

        # Run the server with SSL
        logger.info(f"🖥️▶️ Starting server with SSL (cert: {cert_file}, key: {key_file}).")
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            log_config=None,
            ssl_certfile=cert_file,
            ssl_keyfile=key_file,
        )

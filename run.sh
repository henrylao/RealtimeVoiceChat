#!/bin/bash
# Native run wrapper: external LLM backend (llama-server default) + Kokoro, sm75/CUDA 12.6 torch.
set -e
cd "$(dirname "$0")"
# speaker_id/ prototype stays standalone; expose it for optional live scoring.
export PYTHONPATH="$PWD/speaker_id${PYTHONPATH:+:$PYTHONPATH}"
# Local CUDA 12 system libs not on the loader path: cuDNN 8 for
# ctranslate2/faster-whisper (torch ships cuDNN 9) + portaudio for PyAudio.
export LD_LIBRARY_PATH="$PWD/.libs/cudnn8/lib:$PWD/.libs/portaudio/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
export LLAMA_BASE_URL="${LLAMA_BASE_URL:-http://127.0.0.1:8081/v1}"
export LLM_MODEL="${LLM_MODEL:-daredevil-8b}"
export LLM_PROVIDER="${LLM_PROVIDER:-llamacpp}"
export TTS_ENGINE="${TTS_ENGINE:-kokoro}"
# Opt-in per-turn audio collection for development (default on during dev phase).
export AUDIO_DUMP="${AUDIO_DUMP:-true}"
# Live speaker scoring: paused by default during latency investigation.
# Modules stay intact; set SPEAKER_VERIFY=true to re-enable deliberately.
export SPEAKER_VERIFY="${SPEAKER_VERIFY:-false}"

# Persistent per-boot logs (survive reboots, unlike /tmp). Foreground runs
# keep terminal output; backgrounded runs (stdout not a tty) log to a file.
LOGDIR="$PWD/logs"
mkdir -p "$LOGDIR"
if [ ! -t 1 ]; then
  : "${LOGFILE:=$LOGDIR/server-$(date -u +%Y%m%dT%H%M%SZ).log}"
  exec >>"$LOGFILE" 2>&1
  # Retention: keep newest 10 boot logs.
  ls -t "$LOGDIR"/server-2*.log 2>/dev/null | tail -n +11 | xargs -r rm -f
fi

echo "Checking LLM backend '$LLM_PROVIDER' (model: $LLM_MODEL)..."
.venv/bin/python code/check_backend.py --provider "$LLM_PROVIDER" --require-model "$LLM_MODEL"

exec .venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 --app-dir code

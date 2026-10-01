(function() {
  const originalLog = console.log.bind(console);
  console.log = (...args) => {
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    const ss = String(now.getSeconds()).padStart(2, '0');
    const ms = String(now.getMilliseconds()).padStart(3, '0');
    originalLog(
      `[${hh}:${mm}:${ss}.${ms}]`,
      ...args
    );
  };
})();

const statusDiv = document.getElementById("status");
const messagesDiv = document.getElementById("messages");
const speedSlider = document.getElementById("speedSlider");
speedSlider.disabled = true;  // start disabled
const volumeSlider = document.getElementById("volumeSlider");
// Assistant voice volume (0-150%), persisted across sessions. The server
// already peak-limits TTS to -1 dBFS, so this is pure user preference.
let storedVolume = 100;
try {
  const v = parseInt(localStorage.getItem("ttsVolume"), 10);
  if (Number.isFinite(v)) storedVolume = Math.min(150, Math.max(0, v));
} catch (e) { /* private mode etc. — fall back to 100 */ }
volumeSlider.value = storedVolume;

let socket = null;
let audioContext = null;
let mediaStream = null;
let micWorkletNode = null;
let ttsWorkletNode = null;
let ttsGainNode = null;

function applyVolume() {
  const v = parseInt(volumeSlider.value, 10) || 0;
  if (ttsGainNode) ttsGainNode.gain.value = v / 100;
  try { localStorage.setItem("ttsVolume", String(v)); } catch (e) { /* ignore */ }
}
volumeSlider.addEventListener("input", applyVolume);

let isTTSPlaying = false;
let ignoreIncomingTTS = false;
// Set when the server evicts us for a newer client; shown on close.
let sessionTakeover = false;

let chatHistory = [];
let typingUser = "";
let typingAssistant = "";

// Server wire format: TTS PCM16 chunks are always mono at this rate
// (UpsampleOverlap.TARGET_RATE). The playback worklet resamples to the
// actual AudioContext rate, so pitch stays correct on 48 kHz, 44.1 kHz
// (many BT speakers / iPhones) and 16 kHz HFP routes.
const TTS_SOURCE_RATE = 48000;

// --- batching + fixed 8‑byte header setup ---
const BATCH_SAMPLES = 2048;
const HEADER_BYTES  = 8;
const FRAME_BYTES   = BATCH_SAMPLES * 2;
const MESSAGE_BYTES = HEADER_BYTES + FRAME_BYTES;

const bufferPool = [];
let batchBuffer = null;
let batchView = null;
let batchInt16 = null;
let batchOffset = 0;

function initBatch() {
  if (!batchBuffer) {
    batchBuffer = bufferPool.pop() || new ArrayBuffer(MESSAGE_BYTES);
    batchView   = new DataView(batchBuffer);
    batchInt16  = new Int16Array(batchBuffer, HEADER_BYTES);
    batchOffset = 0;
  }
}

function flushBatch() {
  const ts = Date.now() & 0xFFFFFFFF;
  batchView.setUint32(0, ts, false);
  const flags = isTTSPlaying ? 1 : 0;
  batchView.setUint32(4, flags, false);

  socket.send(batchBuffer);

  bufferPool.push(batchBuffer);
  batchBuffer = null;
}

function flushRemainder() {
  if (batchOffset > 0) {
    for (let i = batchOffset; i < BATCH_SAMPLES; i++) {
      batchInt16[i] = 0;
    }
    flushBatch();
  }
}

function initAudioContext() {
  if (!audioContext) {
    // Ask for the server rate where the browser honors it (desktop Chrome/
    // Firefox do; iOS Safari often ignores the hint — the worklet resampler
    // below is the real fix). Falls back to the default route rate.
    try {
      audioContext = new AudioContext({ sampleRate: TTS_SOURCE_RATE, latencyHint: 'interactive' });
    } catch (e) {
      console.warn("AudioContext with explicit sampleRate failed, using default:", e);
      audioContext = new AudioContext();
    }
    if (audioContext.sampleRate !== TTS_SOURCE_RATE) {
      console.warn(
        `AudioContext rate ${audioContext.sampleRate} Hz != TTS source ${TTS_SOURCE_RATE} Hz — ` +
        `worklet will resample (ratio ${(TTS_SOURCE_RATE / audioContext.sampleRate).toFixed(4)}). ` +
        `If you hear slowed/flat audio, this mismatch was previously played 1:1.`
      );
    } else {
      console.log(`AudioContext rate matches TTS source: ${audioContext.sampleRate} Hz`);
    }
  }
}

function base64ToInt16Array(b64) {
  const raw = atob(b64);
  const buf = new ArrayBuffer(raw.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) {
    view[i] = raw.charCodeAt(i);
  }
  return new Int16Array(buf);
}

async function startRawPcmCapture() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: { ideal: 24000 },
        channelCount: 1,
        echoCancellation: true,
        // autoGainControl: true,
        noiseSuppression: true
      }
    });
    mediaStream = stream;
    initAudioContext();
    await audioContext.audioWorklet.addModule('/static/pcmWorkletProcessor.js');
    micWorkletNode = new AudioWorkletNode(audioContext, 'pcm-worklet-processor',
      { processorOptions: { highpassHz: 120 } });

    micWorkletNode.port.onmessage = ({ data }) => {
      const incoming = new Int16Array(data);
      let read = 0;
      while (read < incoming.length) {
        initBatch();
        const toCopy = Math.min(
          incoming.length - read,
          BATCH_SAMPLES - batchOffset
        );
        batchInt16.set(
          incoming.subarray(read, read + toCopy),
          batchOffset
        );
        batchOffset += toCopy;
        read       += toCopy;
        if (batchOffset === BATCH_SAMPLES) {
          flushBatch();
        }
      }
    };

    const source = audioContext.createMediaStreamSource(stream);
    source.connect(micWorkletNode);
    statusDiv.textContent = "Recording...";
  } catch (err) {
    statusDiv.textContent = "Mic access denied.";
    console.error(err);
  }
}

async function setupTTSPlayback() {
  await audioContext.audioWorklet.addModule('/static/ttsPlaybackProcessor.js');
  ttsWorkletNode = new AudioWorkletNode(
    audioContext,
    'tts-playback-processor',
    { processorOptions: { sourceRate: TTS_SOURCE_RATE } }
  );
  // Redundant with processorOptions (covers remotes/caches serving an older
  // worklet file that ignores processorOptions).
  ttsWorkletNode.port.postMessage({ type: 'tts_config', sourceRate: TTS_SOURCE_RATE });
  // User volume stage: worklet (unity) -> gain -> speakers.
  ttsGainNode = audioContext.createGain();
  console.log(
    `TTS playback: source ${TTS_SOURCE_RATE} Hz -> context ${audioContext.sampleRate} Hz ` +
    `(step ${(TTS_SOURCE_RATE / audioContext.sampleRate).toFixed(4)} source samples/output sample)`
  );

  ttsWorkletNode.port.onmessage = (event) => {
    const { type, sourceRate, contextRate, underrunSamples } = event.data;
    if (type === 'ttsPlaybackStarted') {
      console.log(
        `TTS worklet playing (source ${sourceRate || '?'} Hz -> context ${contextRate || audioContext.sampleRate} Hz).`
      );
      if (!isTTSPlaying && socket && socket.readyState === WebSocket.OPEN) {
        isTTSPlaying = true;
        console.log(
          "TTS playback started. Reason: ttsWorkletNode Event ttsPlaybackStarted."
        );
        socket.send(JSON.stringify({ type: 'tts_start' }));
      }
    } else if (type === 'ttsPlaybackStopped') {
      // Underrun telemetry: starved samples this answer (ear truth).
      // Forwarded to the server inside tts_stop for the session log.
      const underruns = underrunSamples || 0;
      const ctxRate = contextRate || (audioContext && audioContext.sampleRate) || 0;
      if (underruns > 0) {
        console.warn(
          `TTS playback starved ${underruns} samples (~${(underruns / (ctxRate || 48000) * 1000).toFixed(0)} ms of gaps) this answer.`
        );
      }
      if (isTTSPlaying && socket && socket.readyState === WebSocket.OPEN) {
        isTTSPlaying = false;
        console.log(
          "TTS playback stopped. Reason: ttsWorkletNode Event ttsPlaybackStopped."
        );
        socket.send(JSON.stringify({ type: 'tts_stop', underruns, context_rate: ctxRate }));
      }
    }
  };
  ttsWorkletNode.connect(ttsGainNode);
  ttsGainNode.connect(audioContext.destination);
  applyVolume();
  console.log(`TTS volume: ${volumeSlider.value}%`);
}

function cleanupAudio() {
  if (micWorkletNode) {
    micWorkletNode.disconnect();
    micWorkletNode = null;
  }
  if (ttsWorkletNode) {
    ttsWorkletNode.disconnect();
    ttsWorkletNode = null;
  }
  if (ttsGainNode) {
    ttsGainNode.disconnect();
    ttsGainNode = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getAudioTracks().forEach(track => track.stop());
    mediaStream = null;
  }
}

function renderMessages() {
  messagesDiv.innerHTML = "";
  chatHistory.forEach(msg => {
    const bubble = document.createElement("div");
    bubble.className = `bubble ${msg.role}`;
    bubble.textContent = msg.content;
    messagesDiv.appendChild(bubble);
  });
  if (typingUser) {
    const typing = document.createElement("div");
    typing.className = "bubble user typing";
    typing.innerHTML = typingUser + '<span style="opacity:.6;">✏️</span>';
    messagesDiv.appendChild(typing);
  }
  if (typingAssistant) {
    const typing = document.createElement("div");
    typing.className = "bubble assistant typing";
    typing.innerHTML = typingAssistant + '<span style="opacity:.6;">✏️</span>';
    messagesDiv.appendChild(typing);
  }
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function handleJSONMessage({ type, content, sample_rate }) {
  if (type === "tts_config" && content && content.tts_source_rate) {
    // Server announces its TTS wire rate (currently always 48000).
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage({ type: 'tts_config', sourceRate: content.tts_source_rate });
    }
    return;
  }
  if (type === "partial_user_request") {
    typingUser = content?.trim() ? escapeHtml(content) : "";
    renderMessages();
    return;
  }
  if (type === "final_user_request") {
    if (content?.trim()) {
      chatHistory.push({ role: "user", content, type: "final" });
    }
    typingUser = "";
    renderMessages();
    return;
  }
  if (type === "partial_assistant_answer") {
    typingAssistant = content?.trim() ? escapeHtml(content) : "";
    renderMessages();
    return;
  }
  if (type === "final_assistant_answer") {
    if (content?.trim()) {
      chatHistory.push({ role: "assistant", content, type: "final" });
    }
    typingAssistant = "";
    renderMessages();
    return;
  }
  if (type === "tts_chunk") {
    if (ignoreIncomingTTS) return;
    // Newer servers attach sample_rate; if it ever differs from our assumed
    // TTS_SOURCE_RATE, reconfigure the worklet instead of mis-playing.
    if (sample_rate && sample_rate !== TTS_SOURCE_RATE && ttsWorkletNode) {
      console.warn(`TTS chunk rate ${sample_rate} Hz != assumed ${TTS_SOURCE_RATE} Hz — reconfiguring worklet.`);
      ttsWorkletNode.port.postMessage({ type: 'tts_config', sourceRate: sample_rate });
    }
    const int16Data = base64ToInt16Array(content);
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage(int16Data);
    }
    return;
  }
  if (type === "tts_interruption") {
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage({ type: "clear" });
    }
    isTTSPlaying = false;
    ignoreIncomingTTS = false;
    return;
  }
  if (type === "tts_end") {
    // Server finished this generation: drain remaining buffer, then stop.
    // Starvation without this message never stops playback (jitter buffer).
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage({ type: "tts_end" });
    }
    return;
  }
  if (type === "session_takeover") {
    sessionTakeover = true;
    statusDiv.textContent = content || "Session taken over by another device. Press Start to reclaim.";
    console.log("Session takeover:", content);
    return;
  }
  if (type === "stop_tts") {
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage({ type: "clear" });
    }
    isTTSPlaying = false;
    ignoreIncomingTTS = true;
    console.log("TTS playback stopped. Reason: tts_interruption.");
    socket.send(JSON.stringify({ type: 'tts_stop' }));
    return;
  }
}

function escapeHtml(str) {
  return (str ?? '')
    .replace(/&/g, "&amp;")
    .replace(/</g, "<")
    .replace(/>/g, ">")
    .replace(/"/g, "&quot;");
}

// UI Controls

document.getElementById("clearBtn").onclick = () => {
  chatHistory = [];
  typingUser = typingAssistant = "";
  renderMessages();
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'clear_history' }));
  }
};

speedSlider.addEventListener("input", (e) => {
  const speedValue = parseInt(e.target.value);
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({
      type: 'set_speed',
      speed: speedValue
    }));
  }
  console.log("Speed setting changed to:", speedValue);
});

document.getElementById("startBtn").onclick = async () => {
  sessionTakeover = false;
  if (socket && socket.readyState === WebSocket.OPEN) {
    statusDiv.textContent = "Already recording.";
    return;
  }
  statusDiv.textContent = "Initializing connection...";

  const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  socket = new WebSocket(`${wsProto}//${location.host}/ws`);

  socket.onopen = async () => {
    statusDiv.textContent = "Connected. Activating mic and TTS…";
    await startRawPcmCapture();
    // Tell the server the actual capture rate: browsers (notably iOS)
    // ignore the requested sampleRate, and the AudioContext runs at the
    // hardware rate. The server resamples from this rate to 16 kHz.
    if (socket.readyState === WebSocket.OPEN && audioContext) {
      socket.send(JSON.stringify({
        type: "sample_rate",
        sample_rate: audioContext.sampleRate
      }));
      console.log("Reported capture sample rate:", audioContext.sampleRate);
    }
    await setupTTSPlayback();
    // Debug aid: ?tonetest=440 plays a 2 s 440 Hz sine through the worklet
    // at TTS_SOURCE_RATE. Verify with a tuner app on the BT speaker: if it
    // reads 440 Hz, the resampler is correct; ~405 Hz means 48 kHz data is
    // being played at 44.1 kHz without resampling (old bug).
    maybePlayToneTest();
    speedSlider.disabled = false; 
  };

  socket.onmessage = (evt) => {
    if (typeof evt.data === "string") {
      try {
        const msg = JSON.parse(evt.data);
        handleJSONMessage(msg);
      } catch (e) {
        console.error("Error parsing message:", e);
      }
    }
  };

  socket.onclose = () => {
    if (sessionTakeover) {
      statusDiv.textContent = "Session taken over by another device. Press Start to reclaim.";
      sessionTakeover = false;
    } else {
      statusDiv.textContent = "Connection closed.";
    }
    flushRemainder();
    cleanupAudio();
    speedSlider.disabled = true;
  };

  socket.onerror = (err) => {
    statusDiv.textContent = "Connection error.";
    cleanupAudio();
    console.error(err);
    speedSlider.disabled = true; 
  };
};

document.getElementById("stopBtn").onclick = () => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    flushRemainder();
    socket.close();
  }
  cleanupAudio();
  statusDiv.textContent = "Stopped.";
};

document.getElementById("copyBtn").onclick = () => {
  const text = chatHistory
    .map(msg => `${msg.role.charAt(0).toUpperCase() + msg.role.slice(1)}: ${msg.content}`)
    .join('\n');

  navigator.clipboard.writeText(text)
    .then(() => console.log("Conversation copied to clipboard"))
    .catch(err => console.error("Copy failed:", err));
};

// Debug aid: ?tonetest=<hz> (e.g. ?tonetest=440). Synthesizes 2 s of sine
// at TTS_SOURCE_RATE and pushes it through the same worklet path as tts_chunk.
function maybePlayToneTest() {
  let freq = 0;
  try {
    freq = parseFloat(new URLSearchParams(window.location.search).get('tonetest'));
  } catch (e) { /* ignore */ }
  if (!freq || !(freq > 20 && freq < 8000) || !ttsWorkletNode) return;
  const seconds = 2;
  const n = Math.floor(TTS_SOURCE_RATE * seconds);
  const tone = new Int16Array(n);
  for (let i = 0; i < n; i++) {
    tone[i] = Math.floor(32767 * 0.5 * Math.sin(2 * Math.PI * freq * i / TTS_SOURCE_RATE));
  }
  console.log(`Tone test: ${freq} Hz, ${seconds}s at ${TTS_SOURCE_RATE} Hz source rate -> context ${audioContext.sampleRate} Hz. Expect ${freq} Hz on a tuner.`);
  const CHUNK = 4096;
  for (let off = 0; off < n; off += CHUNK) {
    ttsWorkletNode.port.postMessage(tone.subarray(off, Math.min(off + CHUNK, n)));
  }
  ttsWorkletNode.port.postMessage({ type: 'tts_end' });
}

// First render
renderMessages();

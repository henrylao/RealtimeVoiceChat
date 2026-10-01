class TTSPlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // Wire format contract: the server always sends mono PCM16 at this rate
    // (see UpsampleOverlap.TARGET_RATE in upsample_overlap.py). The worklet
    // resamples to the *context* rate (`sampleRate`) so playback pitch is
    // correct on every route: built-in speaker, headphones, BT speaker, ...
    // Without this, 48 kHz data played 1:1 at a 44.1 kHz context sounds
    // ~8% slow/flat, and at a 16 kHz HFP context sounds 3x slow.
    const opts = (options && options.processorOptions) || {};
    this.SOURCE_RATE = opts.sourceRate || 48000;

    // Source-sample FIFO (float32 mono, SOURCE_RATE). Incoming Int16 chunks
    // are converted on arrival; the render loop consumes them fractionally.
    this.srcQueue = [];
    this.srcLen = 0; // total buffered source samples
    this.srcPos = 0; // fractional read position in source samples
    this.prevSample = 0; // last emitted output sample (for PLC fade)
    this.samplesRemaining = 0; // == srcLen, kept for gate readability

    this.isPlaying = false;
    // Pre-buffer gate in *source* samples: 500 ms of audio regardless of
    // device rate. (The old code compared device-rate prebuffer against
    // source-rate counts, under-buffering on 44.1k/16k routes.)
    this.PREBUFFER_SOURCE = Math.floor(this.SOURCE_RATE * 0.5);
    // Set by explicit 'tts_end' from the server (generation complete).
    // Starvation (empty buffer, no end) is NOT completion: keep playing
    // state, bridge with faded PLC, and wait for more audio.
    this.expectingEnd = false;
    // Packet-loss concealment: linear fade from the last real sample over
    // FADE_SAMPLES instead of hard digital zeros (click/pop source).
    this.FADE_SAMPLES = 128;
    this.fadeLeft = 0;
    this.lastValue = 0;
    // Underrun telemetry: frames emitted without real audio (faded PLC or
    // silence) while playing. Reported at stop so the server log shows what
    // the ear actually heard, independent of send-side timing.
    this.underrunSamples = 0;

    // Fractional step: source samples consumed per output sample.
    this.step = this.SOURCE_RATE / sampleRate;

    // Listen for incoming messages
    this.port.onmessage = (event) => {
      const data = event.data;
      if (data && typeof data === "object" && !(data instanceof Int16Array)) {
        if (data.type === "clear") {
          // Clear the TTS buffer and reset playback state.
          this.srcQueue = [];
          this.srcLen = 0;
          this.srcPos = 0;
          this.samplesRemaining = 0;
          this.isPlaying = false;
          this.expectingEnd = false;
          this.fadeLeft = 0;
          this.lastValue = 0;
          this.prevSample = 0;
          this.underrunSamples = 0;
          return;
        }
        if (data.type === "tts_end") {
          // Server finished this generation; drain remaining buffer, then stop.
          this.expectingEnd = true;
          return;
        }
        if (data.type === "tts_config" && data.sourceRate) {
          const rate = Math.floor(data.sourceRate);
          if (rate >= 8000 && rate <= 96000 && rate !== this.SOURCE_RATE) {
            this.SOURCE_RATE = rate;
            this.PREBUFFER_SOURCE = Math.floor(rate * 0.5);
            this.step = rate / sampleRate;
            // Buffered audio is in the old rate; drop it to avoid garbage.
            this.srcQueue = [];
            this.srcLen = 0;
            this.srcPos = 0;
            this.samplesRemaining = 0;
          }
          return;
        }
      }

      // Otherwise assume it's a PCM chunk (e.g., an Int16Array) at SOURCE_RATE.
      if (data instanceof Int16Array) {
        const f = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) {
          f[i] = data[i] / 32768;
        }
        this.srcQueue.push(f);
        this.srcLen += f.length;
        this.samplesRemaining = this.srcLen;
      }
    };
  }

  // Random access into the FIFO without shifting (O(chunks), chunks are few).
  _srcGet(idx) {
    let i = idx;
    for (let q = 0; q < this.srcQueue.length; q++) {
      const c = this.srcQueue[q];
      if (i < c.length) return c[i];
      i -= c.length;
    }
    return 0;
  }

  // Drop the first n source samples after they are fully consumed.
  _srcDrop(n) {
    let left = n;
    while (left > 0 && this.srcQueue.length > 0) {
      const c = this.srcQueue[0];
      if (c.length <= left) {
        this.srcQueue.shift();
        left -= c.length;
        this.srcLen -= c.length;
      } else {
        this.srcQueue[0] = c.subarray(left);
        this.srcLen -= left;
        left = 0;
      }
    }
    this.samplesRemaining = this.srcLen;
  }

  process(inputs, outputs) {
    const out = outputs[0];
    const nFrames = out[0].length;
    const nChannels = out.length;

    // Start gate: wait for a full pre-buffer, unless the stream already
    // ended with a short reply (< pre-buffer of audio banked).
    if (!this.isPlaying) {
      const ready = this.srcLen >= this.PREBUFFER_SOURCE ||
        (this.expectingEnd && this.srcLen > 0);
      if (!ready) {
        for (let ch = 0; ch < nChannels; ch++) out[ch].fill(0);
        // Edge: end arrived with no audio at all (empty reply) — finish now
        // so we never hang in a pre-play state.
        if (this.expectingEnd && this.srcLen === 0) {
          this.expectingEnd = false;
          this.port.postMessage({ type: 'ttsPlaybackStopped' });
        }
        return true;
      }
      this.isPlaying = true;
      this.underrunSamples = 0; // per-answer counter
      this.port.postMessage({
        type: 'ttsPlaybackStarted',
        sourceRate: this.SOURCE_RATE,
        contextRate: sampleRate,
      });
    }

    for (let f = 0; f < nFrames; f++) {
      const idx = Math.floor(this.srcPos);
      const frac = this.srcPos - idx;
      let sample;
      let haveAudio = false;
      // Need s(idx) and s(idx+1) for linear interpolation.
      if (idx + 1 < this.srcLen) {
        const s0 = this._srcGet(idx);
        const s1 = this._srcGet(idx + 1);
        sample = s0 + (s1 - s0) * frac;
        haveAudio = true;
      } else if (idx < this.srcLen) {
        // Last buffered sample: hold it (no next sample to interp to yet).
        sample = this._srcGet(idx);
        haveAudio = true;
      } else {
        // Underrun: faded PLC while starved, plain silence after fade.
        // Never signals stop here — only an explicit end or clear does that.
        // Counted for telemetry (reported at natural end below).
        if (this.isPlaying) this.underrunSamples++;
        sample = 0;
        if (this.fadeLeft > 0) {
          sample = this.lastValue * (this.fadeLeft / this.FADE_SAMPLES);
          this.fadeLeft--;
        }
      }
      if (haveAudio) {
        this.srcPos += this.step;
        this.lastValue = sample;
        this.fadeLeft = this.FADE_SAMPLES; // fresh audio replenishes fade budget
      }
      this.prevSample = sample;
      for (let ch = 0; ch < nChannels; ch++) {
        out[ch][f] = sample;
      }
    }

    // Drop fully-consumed integer source samples to bound memory.
    // No drift: fractional remainder stays in srcPos.
    const consumed = Math.floor(this.srcPos);
    if (consumed > 0) {
      const drop = Math.min(consumed, this.srcLen);
      this._srcDrop(drop);
      this.srcPos -= drop;
      // If we consumed everything but fractional pos pointed past the end
      // (short final chunk), clamp to avoid runaway.
      if (this.srcLen === 0) this.srcPos = 0;
    }

    // Natural end: buffer drained after explicit tts_end and fade played out.
    if (this.srcLen === 0 && this.expectingEnd && this.fadeLeft === 0) {
      this.isPlaying = false;
      this.expectingEnd = false;
      this.port.postMessage({
        type: 'ttsPlaybackStopped',
        underrunSamples: this.underrunSamples,
        contextRate: sampleRate,
      });
    }

    return true;
  }
}

registerProcessor('tts-playback-processor', TTSPlaybackProcessor);

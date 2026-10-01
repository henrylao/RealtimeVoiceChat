// static/pcmWorkletProcessor.js
class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // One-pole highpass on the mic path: desk bumps, handling rumble and
    // mic-pop energy live mostly below ~150 Hz; speech intelligibility does
    // not. Attenuating the sub-bass before VAD/STT cuts false triggers from
    // rough device handling with no added latency and negligible voice impact.
    const opts = (options && options.processorOptions) || {};
    this.cutoffHz = opts.highpassHz || 120;
    // One-pole HP: y[i] = a * (y[i-1] + x[i] - x[i-1]), a = e^(-2*pi*fc/fs).
    this.hpA = Math.exp(-2 * Math.PI * this.cutoffHz / sampleRate);
    this.hpPrevX = 0;
    this.hpPrevY = 0;
  }

  process(inputs) {
    const in32 = inputs[0][0];
    if (in32) {
      // convert Float32 → highpassed Int16 in the worklet
      const int16 = new Int16Array(in32.length);
      const a = this.hpA;
      let px = this.hpPrevX;
      let py = this.hpPrevY;
      for (let i = 0; i < in32.length; i++) {
        const x = in32[i];
        const y = a * (py + x - px);
        px = x;
        py = y;
        let s = y;
        s = s < -1 ? -1 : s > 1 ? 1 : s;
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      this.hpPrevX = px;
      this.hpPrevY = py;
      // send raw ArrayBuffer, transferable
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }
    return true;
  }
}

registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);

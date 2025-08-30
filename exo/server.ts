import express from "express";

const app = express();
const PORT = 3000;

const SIZE_MB = 4;
const NUM_FLOATS = (SIZE_MB * 1024 * 1024) / 4;
var fakeData: ArrayBuffer;

function generateFakeData() {
  const buffer = new ArrayBuffer(NUM_FLOATS * 4);
  const view = new Float32Array(buffer);

  // Parameters for fake broadband sinewave signal
  const numSines = 5;
  const baseFreqs = [0.05, 0.11, 0.21, 0.31, 0.42]; // relative frequencies
  const amplitudes = [1.0, 0.7, 0.5, 0.8, 1.2];
  const drift = 0.00005;

  // Emission lines (normalized to 0..1, like your FFT bin space)
  const emissionBins = [0.278, 0.31, 0.338]; // fake positions for HI and OH lines
  const emissionWidth = 0.0005;

  for (let i = 0; i < NUM_FLOATS; i++) {
    let t = i;
    let sample = 0;

    // Broadband noise with drifting tones
    for (let s = 0; s < numSines; s++) {
      const f = baseFreqs[s] + drift * i;
      sample += amplitudes[s] * Math.sin(2 * Math.PI * f * t);
    }

    // Add sharp spikes as narrowband emissions
    for (let e = 0; e < emissionBins.length; e++) {
      const freq = emissionBins[e];
      // Tiny constant frequency, no drift
      sample += 0.5 * Math.sin(2 * Math.PI * freq * t);
    }

    view[i] = sample / (numSines + emissionBins.length);
  }

  console.log(
    `🚀 Generated ${SIZE_MB}MB of fake signal data with emission lines`
  );

  return buffer;
}

app.use(express.static("dist"));

app.get("/data", (req, res) => {
  if (!fakeData) {
    fakeData = generateFakeData();
  }

  res.setHeader("Content-Type", "application/octet-stream");
  res.setHeader("Content-Length", fakeData.byteLength.toString());
  res.send(Buffer.from(fakeData));
});

app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
});

import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// Serve static files from dist
app.use(express.static(path.join(__dirname, "dist")));

// Fake 256MB data endpoint
const SIZE_MB = 256;
const NUM_FLOATS = (SIZE_MB * 1024 * 1024) / 4;
const buffer = new ArrayBuffer(NUM_FLOATS * 4);
const view = new Float32Array(buffer);
for (let i = 0; i < NUM_FLOATS; i++) view[i] = Math.random();

app.get("/data", (req, res) => {
  res.setHeader("Content-Type", "application/octet-stream");
  res.setHeader("Content-Length", buffer.byteLength.toString());
  res.send(Buffer.from(buffer));
});

app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
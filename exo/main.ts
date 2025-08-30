const CHUNK_SIZE = 32768; // 32k samples per FFT run
const FFT_BINS = 512;

async function fetchData(): Promise<Float32Array> {
  console.log("Fetching data from /data …");
  const res = await fetch("/data");
  const buffer = await res.arrayBuffer();
  console.log(`✅ Got ${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB`);
  return new Float32Array(buffer);
}

async function init() {
  const canvas = document.getElementById("spectrogram") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;
  const width = FFT_BINS;
  const height = 512;
  canvas.width = width;
  canvas.height = height;

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU not supported");
  const device = await adapter.requestDevice();

  // Load WGSL shader
  const shaderModule = device.createShaderModule({
    code: await (await fetch("fft.wgsl")).text(),
  });

  // === FETCH DATA ===
  const data = await fetchData();
  const sampleCount = data.length;
  console.log(`Total samples: ${sampleCount}`);

  // GPU Buffers (small chunks)
  const inputBuffer = device.createBuffer({
    size: CHUNK_SIZE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const outputBuffer = device.createBuffer({
    size: FFT_BINS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Double-buffer readback so we don't stall the GPU
  const readbackBuffers = [0, 1].map(() =>
    device.createBuffer({
      size: FFT_BINS * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
  );

  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  });

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: { module: shaderModule, entryPoint: "main" },
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });

  let offset = 0;
  let yOffset = 0;
  let frame = 0;

  async function render() {
    if (offset + CHUNK_SIZE > sampleCount) offset = 0;

    // Push small chunk into GPU
    device.queue.writeBuffer(
      inputBuffer,
      0,
      data.subarray(offset, offset + CHUNK_SIZE)
    );
    offset += CHUNK_SIZE;

    // Update FFT params
    const startFreq = 0;
    const endFreq = 1024;
    const stride = (endFreq - startFreq) / FFT_BINS;
    device.queue.writeBuffer(
      paramsBuffer,
      0,
      new Float32Array([startFreq, endFreq, stride, CHUNK_SIZE])
    );

    // Pick double-buffer
    const currentReadback = readbackBuffers[frame % 2];

    // Dispatch compute pass
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(FFT_BINS / 64));
    pass.end();

    // Copy FFT results into readback buffer
    encoder.copyBufferToBuffer(
      outputBuffer,
      0,
      currentReadback,
      0,
      FFT_BINS * 4
    );
    device.queue.submit([encoder.finish()]);

    // Wait for GPU to finish copying *only this chunk*
    await currentReadback.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(currentReadback.getMappedRange());

    // Draw row to spectrogram
    const row = ctx.createImageData(width, 1);
    for (let x = 0; x < width; x++) {
      const v = Math.min(255, result[x] * 255); // normalize
      const idx = x * 4;
      row.data[idx + 0] = v;
      row.data[idx + 1] = v;
      row.data[idx + 2] = v;
      row.data[idx + 3] = 255;
    }
    ctx.putImageData(row, 0, yOffset);

    currentReadback.unmap();

    yOffset = (yOffset + 1) % height;
    frame++;

    requestAnimationFrame(render);
  }

  render();
}

init();

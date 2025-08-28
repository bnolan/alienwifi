async function fetchData(): Promise<Float32Array> {
  console.log("Fetching 256MB fake data...");
  const res = await fetch("/data");
  const buffer = await res.arrayBuffer();
  console.log(`✅ Got ${(buffer.byteLength / 1024 / 1024).toFixed(1)}MB`);
  return new Float32Array(buffer);
}

async function init() {
  const webgpuCanvas = document.getElementById(
    "spectrogram"
  ) as HTMLCanvasElement;
  const ctx = webgpuCanvas.getContext("webgpu") as unknown as GPUCanvasContext;

  const debugCanvas = document.getElementById("debug") as HTMLCanvasElement;
  const ctx2d = debugCanvas.getContext("2d")!;

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU not supported");

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });

  ctx.configure({
    device,
    format: navigator.gpu.getPreferredCanvasFormat(),
    alphaMode: "premultiplied",
  });

  const width = 512; // frequency bins
  const height = 512; // time slices

  // Fetch 256MB fake Float32 data
  const data = await fetchData();
  console.log(`Total floats: ${data.length}`);

  // Upload all data into a single storage buffer
  const inputBuffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(inputBuffer.getMappedRange()).set(data);
  inputBuffer.unmap();

  console.log("✅ Uploaded fetched 256MB to GPU");

  // Output buffer for compute results
  const outputBuffer = device.createBuffer({
    size: data.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  const shaderModule = device.createShaderModule({
    code: await (await fetch("fft.wgsl")).text(),
  });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
    ],
  });

  // === CHUNKED COMPUTE DISPATCH ===
  const WORKGROUP_SIZE = 64;
  const CHUNK_SIZE_BYTES = 64 * 1024 * 1024; // 64MB
  const floatsPerChunk = CHUNK_SIZE_BYTES / 4; // Float32 = 4 bytes
  const totalChunks = Math.ceil(data.byteLength / CHUNK_SIZE_BYTES);

  console.log(`Processing in ${totalChunks} chunks of ~64MB each`);

  const encoder = device.createCommandEncoder();

  for (let chunk = 0; chunk < totalChunks; chunk++) {
    const startFloat = chunk * floatsPerChunk;
    const remaining = data.length - startFloat;
    const floatsThisChunk = Math.min(floatsPerChunk, remaining);

    const workgroupsNeeded = Math.ceil(floatsThisChunk / WORKGROUP_SIZE);
    const safeWorkgroups = Math.min(workgroupsNeeded, 65535);

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);

    // Dispatch limited number of workgroups per chunk
    pass.dispatchWorkgroups(safeWorkgroups);
    pass.end();
  }

  device.queue.submit([encoder.finish()]);
  console.log("✅ Finished compute passes");

  // === READ RESULTS BACK ===
  const staging = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const encoder2 = device.createCommandEncoder();
  encoder2.copyBufferToBuffer(outputBuffer, 0, staging, 0, data.byteLength);
  device.queue.submit([encoder2.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange());

  // === VISUALIZE SMALL SUBSET ===
  const imageData = ctx2d.createImageData(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const v = out[idx] * 255;
      const pixel = idx * 4;
      imageData.data[pixel + 0] = v;
      imageData.data[pixel + 1] = v;
      imageData.data[pixel + 2] = v;
      imageData.data[pixel + 3] = 255;
    }
  }
  ctx2d.putImageData(imageData, 0, 0);
  staging.unmap();

  console.log("✅ Visualization done");
}

init();

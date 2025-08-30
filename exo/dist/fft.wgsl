struct Params {
    startFreq: f32,
    endFreq: f32,
    stride: u32,
};

@group(0) @binding(0)
var<storage, read> inputData: array<f32>;

@group(0) @binding(1)
var<storage, read_write> outputData: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let bin = id.x;

    if (bin >= 512u) {
        return;
    }

    let freq = params.startFreq + f32(bin) *
        (params.endFreq - params.startFreq) / 512.0;

    var real = 0.0;
    var imag = 0.0;
    let stride = params.stride;
    let sampleCount = arrayLength(&inputData) / stride;

    for (var n: u32 = 0u; n < 512; n = n + 1u) {
        // let sampleIndex = int(n * stride;
        let sample = inputData[bin + n];
        // let phase = 2.0 * 3.14159265359 * f32(n) * freq / f32(sampleCount);

        real += sample; // * cos(phase);
        imag -= sample; // * sin(phase);
    }

    // Magnitude
    // var mag = sqrt(real * real + imag * imag);

    // 🔹 LOG SCALE FOR CONTRAST
    // let safeMag = max(mag, 1e-9);
    // let logMag = log2(safeMag + 1.0);

    // 🔹 Normalize into [0,1]
    // let scaled = clamp(logMag / 10.0, 0.0, 1.0);

    outputData[bin] = fract(imag + real); // fract(inputData[bin]);
}
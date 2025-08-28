@group(0) @binding(0)
var<storage, read> inputData: array<f32>;

@group(0) @binding(1)
var<storage, read_write> outputData: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;

    if (i >= arrayLength(&inputData)) {
        return;
    }

    // Fake FFT: apply sine modulation
    let v = inputData[i];
    outputData[i] = abs(sin(v * 0.1 + f32(i) * 0.05));
}
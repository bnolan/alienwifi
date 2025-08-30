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

    // Just check we don't overflow the buffer
    if (bin >= 512u) {
        return;
    }

    // 🔹 Pass raw floats directly into output
    // Wrap them into [0,1] so you can see *something*
    var v = inputData[bin];
    v = fract(abs(v) * 1234.567); // Make some noise if data’s real

    outputData[bin] = v;
}
@group(0) @binding(0)
var<storage, read> input: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

@group(0) @binding(2)
var<storage, read_write> workgroup_sums: array<u32>;

var<workgroup> output_shared: array<u32, 512>;

const BLOCK_SIZE = 256u;

// Work-efficient parallel inclusive scan.
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
@compute
@workgroup_size(256)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    // Each block handles BLOCK_SIZE * 2 elements.
    // Each thread loads two elements.
    let len = arrayLength(&input);
    let input_index = workgroup_id.x * BLOCK_SIZE * 2u + local_id.x;
    if (input_index < len) {
        output_shared[local_id.x] = input[input_index];
    } else {
        output_shared[local_id.x] = 0u;
    }
    if (input_index + BLOCK_SIZE < len) {
        output_shared[BLOCK_SIZE + local_id.x] = input[BLOCK_SIZE + input_index];
    } else {
        output_shared[BLOCK_SIZE + local_id.x] = 0u;
    }

    // 1. Parallel Scan Step.
    for (var stride = 1u; stride < 2u * BLOCK_SIZE; stride *= 2u) {
        // At each iteration we produce sums using a larger stride.
        // This produces partial sums needed for step 2.
        // 0: 0+1, 2+3, ...
        // 1: 1+3, 5+7, ...
        // 2: 7+15, ...
        workgroupBarrier();
        let index = (local_id.x + 1u) * stride * 2u - 1u;
        if (index < 2u * BLOCK_SIZE) {
            output_shared[index] += output_shared[index - stride];
        }
    }

    // 2. Post Scan Step (compute final block result).
    for (var stride = BLOCK_SIZE / 2u; stride > 0u; stride /= 2u) {
        // Combine the partial sums into the proper scan values.
        // This is basically the reverse of step 1.
        workgroupBarrier();
        let index = (local_id.x + 1u) * stride * 2u - 1u;
        if (index + stride < 2u * BLOCK_SIZE) {
            output_shared[index + stride] += output_shared[index];
        }
    }

    // 3. Write the final result to global memory.
    // Each thread writes two elements.
    workgroupBarrier();
    let outputIndex = workgroup_id.x * BLOCK_SIZE * 2u + local_id.x;
    if (outputIndex < len) {
        output[outputIndex] = output_shared[local_id.x];
    }
    if (outputIndex + BLOCK_SIZE < len) {
        output[BLOCK_SIZE + outputIndex] = output_shared[BLOCK_SIZE + local_id.x];
    }

    // Save the total sum for the next steps.
    workgroup_sums[workgroup_id.x] = output_shared[2u * BLOCK_SIZE - 1u];
}
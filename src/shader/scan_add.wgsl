@group(0) @binding(0)
var<storage, read_write> output: array<u32>;

@group(0) @binding(1)
var<storage, read_write> workgroup_sums: array<u32>;

const BLOCK_SIZE = 256u;

@compute
@workgroup_size(256)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    // Account for each workgroup processing twice as many elements for scan.
    let index = workgroup_id.x * BLOCK_SIZE * 2u + local_id.x;
    let len = arrayLength(&output);

    if (workgroup_id.x > 0u) {
        // Add the sum of the previous workgroups to each element.
        // The scan is inclusive, so shift the index by 1.
        let previous_sum = workgroup_sums[workgroup_id.x - 1u];
        if (index < len) {
            output[index] += previous_sum;
        }
        if (index + BLOCK_SIZE < len) {
            output[index + BLOCK_SIZE] += previous_sum;
        }
    }
}
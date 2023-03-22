struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    base_vertex: u32,
    base_instance: u32,
}

@group(0)
@binding(0)
var<storage, read_write> draws: array<DrawIndirect>;

// TODO: also pass in bounding information and camera info

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // TODO: modify the draw indirect buffer using culling.
    let i = global_id.x;
    draws[i].instance_count = draws[i].instance_count;
}
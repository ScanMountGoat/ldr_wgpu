use glam::Vec3;

pub fn vertex_normals(vertices: &[Vec3], vertex_indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; vertices.len()];
    for face in vertex_indices.chunks_exact(3) {
        let v1 = vertices[face[0] as usize];
        let v2 = vertices[face[1] as usize];
        let v3 = vertices[face[2] as usize];

        // Don't normalize since the cross product is proportional to face area.
        // This weights the normals by face area when summing later.
        let u = v2 - v1;
        let v = v3 - v1;
        let normal = u.cross(v);

        for i in face {
            normals[*i as usize] += normal;
        }
    }

    for n in &mut normals {
        *n = n.normalize();
    }

    normals
}

use std::collections::BTreeSet;

use glam::Vec3;

// TODO: Add an option to index this separately instead of returning the set?
// i.e. normals + normals indices
pub fn triangle_face_vertex_normals(
    vertices: &[Vec3],
    vertex_indices: &[u32],
) -> (Vec<BTreeSet<usize>>, Vec<Vec3>) {
    // TODO: move this to ldr_tools.
    // TODO: Smooth normals based on hard edges and face angle threshold.
    let face_normals: Vec<_> = vertex_indices
        .chunks_exact(3)
        .map(|face| {
            let v1 = vertices[face[0] as usize];
            let v2 = vertices[face[1] as usize];
            let v3 = vertices[face[2] as usize];

            let u = v2 - v1;
            let v = v3 - v1;
            u.cross(v)
        })
        .collect();

    // Assume the position indices are fully welded.
    // This makes it easy to calculate the indices of adjacent faces for each vertex.
    let mut vertex_adjacent_faces = vec![Vec::new(); vertices.len()];
    for (i, face) in vertex_indices.chunks_exact(3).enumerate() {
        vertex_adjacent_faces[face[0] as usize].push(i);
        vertex_adjacent_faces[face[1] as usize].push(i);
        vertex_adjacent_faces[face[2] as usize].push(i);
    }

    // Use a BTreeSet for a consistent hash value.
    let filtered_adjacent_faces: Vec<BTreeSet<_>> = vertex_indices
        .iter()
        .enumerate()
        .map(|(i, vertex_index)| {
            // TODO: Also check hard edges.
            let face_index = i / 3;
            let face_normal = face_normals[face_index];
            vertex_adjacent_faces[*vertex_index as usize]
                .iter()
                .copied()
                .filter(|f| face_normals[*f].angle_between(face_normal).abs() < 89f32.to_radians())
                .collect()
        })
        .collect();

    let face_vertex_normals: Vec<_> = filtered_adjacent_faces
        .iter()
        .map(|faces| {
            // TODO: Optimize this?
            // TODO: Add to geometry_tools?
            // Smooth normals are the average of the adjacent face normals.
            // TODO: Weight by face area?
            faces
                .iter()
                .map(|f| face_normals[*f])
                .sum::<Vec3>()
                .normalize()
        })
        .collect();
    (filtered_adjacent_faces, face_vertex_normals)
}

#[cfg(test)]
mod tests {
    use super::*;

    use glam::vec3;

    #[test]
    fn normals_single_triangle() {
        let (adjacent, normals) = triangle_face_vertex_normals(
            &[
                vec3(-5f32, 5f32, 1f32),
                vec3(-5f32, 0f32, 1f32),
                vec3(0f32, 0f32, 1f32),
            ],
            &[0, 1, 2],
        );

        assert_eq!(vec![BTreeSet::from([0]); 3], adjacent);
        assert_eq!(vec![vec3(0.0, 0.0, 1.0); 3], normals);
    }

    // TODO: Test a simple shape with and without hard edges
}

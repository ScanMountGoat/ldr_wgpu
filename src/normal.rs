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

            // Don't normalize since the cross product is proportional to face area.
            // This weights the normals by face area when summing later.
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
    // Use a large angle threshold to only add creases on extreme angle changes.
    let filtered_adjacent_faces: Vec<BTreeSet<_>> = vertex_indices
        .iter()
        .enumerate()
        .map(|(i, vertex_index)| {
            let face_index = i / 3;
            let face_normal = face_normals[face_index];
            vertex_adjacent_faces[*vertex_index as usize]
                .iter()
                .copied()
                .filter(|f| face_normals[*f].angle_between(face_normal).abs() < 90f32.to_radians())
                .collect()
        })
        .collect();

    let face_vertex_normals: Vec<_> = filtered_adjacent_faces
        .iter()
        .map(|faces| {
            // TODO: Optimize this?
            // TODO: Add to geometry_tools?
            // Smooth normals are the average of the adjacent face normals.
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

    fn set<const N: usize>(x: [usize; N]) -> BTreeSet<usize> {
        BTreeSet::from(x)
    }

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

        assert_eq!(vec![set([0]); 3], adjacent);
        assert_eq!(vec![vec3(0.0, 0.0, 1.0); 3], normals);
    }

    #[test]
    fn normals_tetrahedron() {
        // TODO: Make this more mathematically precise
        let (adjacent, normals) = triangle_face_vertex_normals(
            &[
                vec3(0.000000, -0.707000, -1.000000),
                vec3(0.866025, -0.707000, 0.500000),
                vec3(-0.866025, -0.707000, 0.500000),
                vec3(0.000000, 0.707000, 0.000000),
            ],
            &[0, 3, 1, 0, 1, 2, 1, 3, 2, 2, 3, 0],
        );
        // The angle threshold should split all faces.
        assert_eq!(
            vec![
                set([0]),
                set([0]),
                set([0]),
                set([1]),
                set([1]),
                set([1]),
                set([2]),
                set([2]),
                set([2]),
                set([3]),
                set([3]),
                set([3])
            ],
            adjacent
        );
        let n0 = vec3(0.816483, 0.333378, -0.47139645);
        let n1 = vec3(0.0, -1.0, 0.0);
        let n2 = vec3(0.0, 0.3333781, 0.94279325);
        let n3 = vec3(-0.816483, 0.333378, -0.47139645);
        assert_eq!(
            vec![n0, n0, n0, n1, n1, n1, n2, n2, n2, n3, n3, n3],
            normals
        );
    }

    // TODO: Test a simple 2D mesh with and without hard edges
}

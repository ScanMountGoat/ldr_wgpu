use std::collections::{BTreeSet, HashMap, HashSet};

/// Calculate new vertices and indices by splitting the edges in `sharp_edges`.
/// This works similarly to Blender's "edge split" for calculating normals.
// https://github.com/blender/blender/blob/a32dbb8/source/blender/geometry/intern/mesh_split_edges.cc
pub fn split_sharp_edges<T: Copy>(
    vertices: &[T],
    vertex_indices: &[u32],
    sharp_edges: &[[u32; 2]],
) -> (Vec<T>, Vec<u32>) {
    // TODO: should ldr_tools just store sharp edges?
    // TODO: separate function and tests for this?
    // Mark any vertices on a sharp edge as sharp to duplicate later.
    let mut is_sharp_vertex = vec![false; vertices.len()];
    let mut sharp_undirected_edges = HashSet::new();
    for [v0, v1] in sharp_edges {
        // Treat edges as undirected.
        sharp_undirected_edges.insert([*v0, *v1]);
        sharp_undirected_edges.insert([*v1, *v0]);

        is_sharp_vertex[*v0 as usize] = true;
        is_sharp_vertex[*v1 as usize] = true;
    }

    let adjacent_faces = adjacent_faces(vertices, vertex_indices);

    let (split_vertices, mut split_vertex_indices, duplicate_edges) =
        split_face_verts(vertices, vertex_indices, &adjacent_faces, &is_sharp_vertex);

    merge_duplicate_edges(
        &mut split_vertex_indices,
        vertex_indices,
        duplicate_edges,
        sharp_undirected_edges,
        adjacent_faces,
    );

    reindex_vertices(split_vertex_indices, split_vertices)
}

fn reindex_vertices<T: Copy>(
    split_vertex_indices: Vec<u32>,
    split_vertices: Vec<T>,
) -> (Vec<T>, Vec<u32>) {
    // Reindex to use the indices 0..N.
    // Truncate the split vertices to length N.
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    let mut remapped_indices = HashMap::new();

    // Map each index to a new index.
    // Use this mapping to create the new vertices as well.
    for index in split_vertex_indices {
        if let Some(new_index) = remapped_indices.get(&index) {
            indices.push(*new_index);
        } else {
            let new_index = remapped_indices.len() as u32;
            verts.push(split_vertices[index as usize]);
            indices.push(new_index);
            remapped_indices.insert(index, new_index);
        }
    }

    (verts, indices)
}

fn adjacent_faces<T>(vertices: &[T], vertex_indices: &[u32]) -> Vec<BTreeSet<usize>> {
    // TODO: Function and tests for this since it's shared?
    // TODO: Should this use the old non duplicated indices?
    // Assume the position indices are fully welded.
    // This simplifies calculating the adjacent face indices for each vertex.
    let mut adjacent_faces = vec![BTreeSet::new(); vertices.len()];
    for (i, face) in vertex_indices.chunks_exact(3).enumerate() {
        adjacent_faces[face[0] as usize].insert(i);
        adjacent_faces[face[1] as usize].insert(i);
        adjacent_faces[face[2] as usize].insert(i);
    }
    adjacent_faces
}

fn merge_duplicate_edges(
    split_vertex_indices: &mut [u32],
    vertex_indices: &[u32],
    duplicate_edges: Vec<[u32; 2]>,
    sharp_undirected_edges: HashSet<[u32; 2]>,
    adjacent_faces: Vec<BTreeSet<usize>>,
) {
    // Merge any of the duplicated edges that is not marked sharp.
    // We "merge" edges by ensuring they use the same vertex indices.
    // TODO: Double check if there is redundant adjacency calculations with normals later.
    // TODO: Just check duplicated edges here?
    for [v0, v1] in duplicate_edges
        .into_iter()
        .filter(|e| !sharp_undirected_edges.contains(e))
    {
        // Find the two faces indicent to this edge from the old indexing.
        let mut faces = adjacent_faces[v0 as usize].intersection(&adjacent_faces[v1 as usize]);

        // Skip any edges without two incident faces like boundary edges.
        if let (Some(f0), Some(f1)) = (faces.next(), faces.next()) {
            // "merge" this edge by using the F0 indices for the edge in F1
            // The edges use the old indices that haven't been duplicated.
            // This takes advantage of duplicate vertices not increasing the length of the face list.
            // TODO: does this create redundant work?
            // TODO: is it ok to always use the old non duplicated indices here?
            let mut v0_f0 = v0;
            let mut v1_f0 = v1;
            for i in f0 * 3..f0 * 3 + 3 {
                // TODO: function and tests for this?
                if vertex_indices[i] == v0 {
                    v0_f0 = split_vertex_indices[i];
                }
                if vertex_indices[i] == v1 {
                    v1_f0 = split_vertex_indices[i];
                }
            }
            // Apply the first face's edge indices to the second.
            for i in f1 * 3..f1 * 3 + 3 {
                // TODO: match statement.
                // TODO: function and tests for this?
                if vertex_indices[i] == v0 {
                    split_vertex_indices[i] = v0_f0;
                }
                if vertex_indices[i] == v1 {
                    split_vertex_indices[i] = v1_f0;
                }
            }
        }
    }
}

fn split_face_verts<T: Copy>(
    vertices: &[T],
    vertex_indices: &[u32],
    adjacent_faces: &[BTreeSet<usize>],
    is_sharp_vertex: &[bool],
) -> (Vec<T>, Vec<u32>, Vec<[u32; 2]>) {
    // Split sharp edges by duplicating the vertices.
    // This creates some duplicate edges to be cleaned up later.
    let mut split_vertices = vertices.to_vec();
    let mut split_vertex_indices = vertex_indices.to_vec();

    let mut duplicate_edges = Vec::new();

    // TODO: Avoid splitting a face vertex more than once?
    // Iterate over all the indices of vertices marked as sharp.
    for vertex_index in is_sharp_vertex
        .iter()
        .enumerate()
        .filter_map(|(v, sharp)| sharp.then_some(v))
    {
        // Duplicate the vertex in all faces except the first.
        // The first face can just use the original index.
        for f in adjacent_faces[vertex_index].iter().skip(1) {
            let face = &mut split_vertex_indices[f * 3..f * 3 + 3];

            // TODO: Find a cleaner way to calculate edges.
            let mut i_face_vert = 0;
            for (j, face_vert) in face.iter_mut().enumerate() {
                if *face_vert == vertex_index as u32 {
                    *face_vert = split_vertices.len() as u32;
                    split_vertices.push(split_vertices[vertex_index]);

                    i_face_vert = j;
                }
            }

            // Track any edges that have been duplicated.
            // The non sharp duplicated edges will be merged later.
            // Take advantage of every vertex being connected in a triangle.
            // Use the original vertices sharp edges refer to the original indices.
            let original_face = &vertex_indices[f * 3..f * 3 + 3];
            let e0 = [
                original_face[i_face_vert],
                original_face[(i_face_vert + 1) % 3],
            ];
            duplicate_edges.push(e0);

            let e1 = [
                original_face[i_face_vert],
                original_face[(i_face_vert + 2) % 3],
            ];
            duplicate_edges.push(e1);
        }
    }

    (split_vertices, split_vertex_indices, duplicate_edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_sharp_edges_triangle_no_sharp_edges() {
        // 2
        // | \
        // 0 - 1

        assert_eq!(
            (vec![0.0, 1.0, 2.0], vec![0, 1, 2]),
            split_sharp_edges(&[0.0, 1.0, 2.0], &[0, 1, 2], &[])
        );
    }

    #[test]
    fn split_sharp_edges_quad() {
        // Quad of two tris and one sharp edge.
        // The topology shouldn't change since 2-3 is already a boundary.
        // 2 - 3
        // | \ |
        // 0 - 1

        let indices = vec![0, 1, 2, 2, 1, 3];
        assert_eq!(
            (vec![0.0, 1.0, 2.0, 3.0], indices.clone()),
            split_sharp_edges(&[0.0, 1.0, 2.0, 3.0], &indices, &[[2, 3]])
        );
    }

    #[test]
    fn split_sharp_edges_two_quads() {
        // Two quads of two tris and twp sharp edges.
        // The topology shouldn't change.
        // 2 - 3 - 5
        // | \ | \ |
        // 0 - 1 - 4

        let indices = vec![0, 1, 2, 2, 1, 3, 3, 1, 4, 3, 4, 5];
        assert_eq!(
            (vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], indices.clone()),
            split_sharp_edges(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &indices, &[[2, 3], [3, 5]])
        );
    }

    #[test]
    fn split_sharp_edges_split_two_quads() {
        // Two quads of two tris and and one sharp edge.
        // 2 - 3 - 4
        // | \ | \ |
        // 0 - 1 - 5

        // The edge 1-3 splits the quads in two.
        // 2 - 3    4 - 7
        // | \ |    | \ |
        // 0 - 1    5 - 6

        let indices = vec![0, 1, 2, 2, 1, 3, 3, 1, 5, 3, 5, 4];
        assert_eq!(
            (
                vec![0.0, 1.0, 2.0, 3.0, 3.0, 1.0, 5.0, 4.0],
                vec![0, 1, 2, 2, 1, 3, 4, 5, 6, 4, 6, 7]
            ),
            split_sharp_edges(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &indices, &[[1, 3]])
        );
    }

    #[test]
    fn split_sharp_edges_split_three_quads() {
        // Three quads of two tris and and one sharp edge.
        // 2 - 3 - 4 - 7
        // | \ | \ | \ |
        // 0 - 1 - 5 - 6

        // The edge 1-3 splits the quads in two.
        // 2 - 3   4 - 7 - 9
        // | \ |   | \ | \ |
        // 0 - 1   5 - 6 - 8

        let indices = vec![0, 1, 2, 2, 1, 3, 3, 1, 5, 3, 5, 4, 4, 5, 6, 4, 6, 7];
        assert_eq!(
            (
                vec![0.0, 1.0, 2.0, 3.0, 3.0, 1.0, 5.0, 4.0, 6.0, 7.0],
                vec![0, 1, 2, 2, 1, 3, 4, 5, 6, 4, 6, 7, 7, 6, 8, 7, 8, 9]
            ),
            split_sharp_edges(
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                &indices,
                &[[1, 3]]
            )
        );
    }

    #[test]
    fn split_sharp_edges_split_four_quads() {
        // Three quads of two tris.
        // This more general case applies to cylinders, cubes, etc.
        // 6 - 7 - 8
        // | \ | \ |
        // 2 - 3 - 5
        // | \ | \ |
        // 0 - 1 - 4

        // After splitting [2,3] and [3,4].
        //  6 - 9 - 11
        //  | \ | \  |
        //  7 - 8 - 10
        //
        //  2 - 3 - 5
        //  | \ | \ |
        //  0 - 1 - 4

        let indices = vec![
            0, 1, 2, 2, 1, 3, 3, 1, 4, 3, 4, 5, 6, 2, 3, 6, 3, 7, 7, 3, 5, 7, 5, 8,
        ];
        assert_eq!(
            (
                vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 7.0, 5.0, 8.0],
                vec![0, 1, 2, 2, 1, 3, 3, 1, 4, 3, 4, 5, 6, 7, 8, 6, 8, 9, 9, 8, 10, 9, 10, 11]
            ),
            split_sharp_edges(
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &indices,
                &[[2, 3], [3, 5]]
            )
        );
    }
}

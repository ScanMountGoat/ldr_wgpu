use std::collections::{BTreeSet, HashSet};

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

    // TODO: Function and tests for this since it's shared?
    // TODO: Should this use the old non duplicated indices?
    // Assume the position indices are fully welded.
    // This makes it easy to calculate the indices of adjacent faces for each vertex.
    let mut vertex_adjacent_faces = vec![BTreeSet::new(); vertices.len()];
    for (i, face) in vertex_indices.chunks_exact(3).enumerate() {
        vertex_adjacent_faces[face[0] as usize].insert(i);
        vertex_adjacent_faces[face[1] as usize].insert(i);
        vertex_adjacent_faces[face[2] as usize].insert(i);
    }

    // Split sharp edges by duplicating the vertices.
    // This creates some duplicate edges to be cleaned up later.
    let mut split_vertices = vertices.to_vec();
    let mut split_vertex_indices = vertex_indices.to_vec();

    let mut duplicate_edges = Vec::new();

    // TODO: Avoid splitting a face vertex more than once?
    // Iterate over all the indices of vertices marked as sharp.
    for (vertex_index, sharp) in is_sharp_vertex.iter().enumerate() {
        if *sharp {
            // Duplicate the vertex in all faces except the first.
            // The first face can just use the original index.
            for f in vertex_adjacent_faces[vertex_index].iter().skip(1) {
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
    }

    // Merge any of the duplicated edges that is not marked sharp.
    // We "merge" edges by ensuring they use the same vertex indices.
    // TODO: Double check if there is redundant adjacency calculations with normals later.
    // TODO: Just check duplicated edges here?
    for [v0, v1] in duplicate_edges
        .into_iter()
        .filter(|e| !sharp_undirected_edges.contains(e))
    {
        // Find the two faces indicent to this edge from the old indexing.
        // TODO: avoid collect
        // dbg!(&vertex_adjacent_faces[v0 as usize]);
        // dbg!(&vertex_adjacent_faces[v1 as usize]);

        let faces: Vec<_> = vertex_adjacent_faces[v0 as usize]
            .intersection(&vertex_adjacent_faces[v1 as usize])
            .collect();
        // dbg!(&faces);
        if faces.len() < 2 {
            // TODO: this should never happen?
            // dbg!(faces);
            continue;
        }
        let f0 = faces[0];
        let f1 = faces[1];

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
    (split_vertices, split_vertex_indices)
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
        // TODO: Should this also merge the position list?
        let indices = vec![0, 1, 2, 2, 1, 3];
        assert_eq!(
            (vec![0.0, 1.0, 2.0, 3.0, 2.0], indices.clone()),
            split_sharp_edges(&[0.0, 1.0, 2.0, 3.0], &indices, &[[2, 3]])
        );
    }

    #[test]
    fn split_sharp_edges_two_quads_no_split() {
        // Two quads of two tris and twp sharp edges.
        // The topology shouldn't change.
        // 2 - 3 - 4
        // | \ | \ |
        // 0 - 1 - 5
        // TODO: Should this also merge the position list?
        let indices = vec![0, 1, 2, 2, 1, 3, 3, 1, 5, 3, 5, 4];
        assert_eq!(
            (
                vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 3.0],
                indices.clone()
            ),
            split_sharp_edges(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &indices, &[[2, 3], [3, 4]])
        );
    }

    #[test]
    fn split_sharp_edges_two_quads_split() {
        // Two quads of two tris and and sharp edges.
        // 2 - 3 - 4
        // | \ | \ |
        // 0 - 1 - 5

        // The edge 1-3 splits the quads in the middle.
        // 2 - 3    8 - 4
        // | \ |    | \ |
        // 0 - 1    7 - 5

        // TODO: Should this also merge the position list?
        let indices = vec![0, 1, 2, 2, 1, 3, 3, 1, 5, 3, 5, 4];
        assert_eq!(
            (
                vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 3.0, 3.0],
                // TODO: have the function reindex to avoid unused indices?
                vec![0, 1, 2, 2, 1, 3, 8, 7, 5, 8, 5, 4]
            ),
            split_sharp_edges(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &indices, &[[1, 3]])
        );
    }

    #[test]
    fn split_sharp_edges_split_two_quads() {
        // Two quads of two tris and and one sharp edge.
        // 2 - 3 - 4
        // | \ | \ |
        // 0 - 1 - 5

        // The edge 1-3 splits the quads in two.
        // 2 - 3    8 - 4
        // | \ |    | \ |
        // 0 - 1    7 - 5

        // TODO: Should this also merge the position list?
        let indices = vec![0, 1, 2, 2, 1, 3, 3, 1, 5, 3, 5, 4];
        assert_eq!(
            (
                vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 3.0, 3.0],
                // TODO: have the function reindex to avoid unused indices?
                vec![0, 1, 2, 2, 1, 3, 8, 7, 5, 8, 5, 4]
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
        // TODO: Is this right?
        // 2 - 3   10 - 4 - 7
        // | \ |    | \ | \ |
        // 0 - 1    9 - 5 - 6

        // TODO: Should this also merge the position list?
        let indices = vec![0, 1, 2, 2, 1, 3, 3, 1, 5, 3, 5, 4, 4, 5, 6, 4, 6, 7];
        assert_eq!(
            (
                vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 1.0, 3.0, 3.0],
                // TODO: have the function reindex to avoid unused indices?
                vec![0, 1, 2, 2, 1, 3, 10, 9, 5, 10, 5, 4, 4, 5, 6, 4, 6, 7]
            ),
            split_sharp_edges(
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                &indices,
                &[[1, 3]]
            )
        );
    }

    // TODO: recreate the incorrect results on cylinders and cones
}

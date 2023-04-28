use std::collections::{BTreeSet, HashMap, HashSet};

/// Calculate new vertices and indices by splitting the edges in `edges`.
/// This works similarly to Blender's "edge split" for calculating normals.
// https://github.com/blender/blender/blob/a32dbb8/source/blender/geometry/intern/mesh_split_edges.cc
pub fn split_edges<T: Copy>(
    vertices: &[T],
    vertex_indices: &[u32],
    edges: &[[u32; 2]],
) -> (Vec<T>, Vec<u32>) {
    // TODO: should ldr_tools just store sharp edges?
    // TODO: separate function and tests for this?
    // Mark any vertices on a sharp edge as sharp to duplicate later.
    let mut is_sharp_vertex = vec![false; vertices.len()];
    let mut sharp_undirected_edges = HashSet::new();
    for [v0, v1] in edges {
        // Treat edges as undirected.
        sharp_undirected_edges.insert([*v0, *v1]);
        sharp_undirected_edges.insert([*v1, *v0]);

        is_sharp_vertex[*v0 as usize] = true;
        is_sharp_vertex[*v1 as usize] = true;
    }

    let old_adjacent_faces = adjacent_faces(vertices, vertex_indices);

    let (split_vertices, mut split_vertex_indices, duplicate_edges) = split_face_verts(
        vertices,
        vertex_indices,
        &old_adjacent_faces,
        &is_sharp_vertex,
    );

    let mut new_adjacent_faces = adjacent_faces(&split_vertices, &split_vertex_indices);

    merge_duplicate_edges(
        &mut split_vertex_indices,
        vertex_indices,
        duplicate_edges,
        sharp_undirected_edges,
        &old_adjacent_faces,
        &mut new_adjacent_faces,
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
    duplicate_edges: HashSet<[u32; 2]>,
    sharp_undirected_edges: HashSet<[u32; 2]>,
    old_adjacent_faces: &[BTreeSet<usize>],
    new_adjacent_faces: &mut [BTreeSet<usize>],
) {
    // Merge any of the duplicated edges that is not marked sharp.
    // We "merge" edges by ensuring they use the same vertex indices.
    // TODO: Double check if there is redundant adjacency calculations with normals later.
    // TODO: Just check duplicated edges here?
    for [v0, v1] in duplicate_edges
        .into_iter()
        .filter(|e| !sharp_undirected_edges.contains(e))
    {
        // Find the faces indicent to this edge from the old indexing.
        let v0_faces = &old_adjacent_faces[v0 as usize];
        let v1_faces = &old_adjacent_faces[v1 as usize];
        let mut faces = v0_faces.intersection(v1_faces);

        if let (Some(f0), Some(f1)) = (faces.next(), faces.next()) {
            // "merge" this edge by using the F0 indices for the edge in F1
            // The edges use the old indices that haven't been duplicated.
            // This takes advantage of duplicate vertices not increasing the length of the face list.
            // TODO: does this create redundant work?
            // TODO: is it ok to always use the old non duplicated indices here?
            let mut v0_f0 = v0;
            let mut v1_f0 = v1;
            // TODO: function to get vertex indices from face?
            for i in f0 * 3..f0 * 3 + 3 {
                // TODO: function and tests for this?
                if vertex_indices[i] == v0 {
                    v0_f0 = split_vertex_indices[i];
                }
                if vertex_indices[i] == v1 {
                    v1_f0 = split_vertex_indices[i];
                }
            }

            let mut v0_f1 = v0;
            let mut v1_f1 = v1;
            // TODO: function to get vertex indices from face?
            for i in f1 * 3..f1 * 3 + 3 {
                // TODO: function and tests for this?
                if vertex_indices[i] == v0 {
                    v0_f1 = split_vertex_indices[i];
                }
                if vertex_indices[i] == v1 {
                    v1_f1 = split_vertex_indices[i];
                }
            }

            // Merge an edge by merging both pairs of verts.
            // The faces adjacent to the f1 verts are now adjacent to the f0 verts.
            new_adjacent_faces[v0_f0 as usize].extend(new_adjacent_faces[v0_f1 as usize].clone());
            new_adjacent_faces[v1_f0 as usize].extend(new_adjacent_faces[v1_f1 as usize].clone());

            // Update the verts in each of the adjacent faces to use the f0 verts.
            // Use the new adjacency to keep track of what has already been merged.
            let v0_faces = &new_adjacent_faces[v0_f0 as usize];
            let v1_faces = &new_adjacent_faces[v1_f0 as usize];
            for adjacent_face in v0_faces.iter().chain(v1_faces.iter()) {
                for i in adjacent_face * 3..adjacent_face * 3 + 3 {
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
}

fn split_face_verts<T: Copy>(
    vertices: &[T],
    vertex_indices: &[u32],
    adjacent_faces: &[BTreeSet<usize>],
    is_sharp_vertex: &[bool],
) -> (Vec<T>, Vec<u32>, HashSet<[u32; 2]>) {
    // Split sharp edges by duplicating the vertices.
    // This creates some duplicate edges to be cleaned up later.
    let mut split_vertices = vertices.to_vec();
    let mut split_vertex_indices = vertex_indices.to_vec();

    let mut duplicate_edges = HashSet::new();

    // Iterate over all the indices of vertices marked as sharp.
    for vertex_index in is_sharp_vertex
        .iter()
        .enumerate()
        .filter_map(|(v, sharp)| sharp.then_some(v))
    {
        for (i_f, f) in adjacent_faces[vertex_index].iter().enumerate() {
            let face = &mut split_vertex_indices[f * 3..f * 3 + 3];

            // TODO: Find a cleaner way to calculate edges.
            let mut i_face_vert = 0;
            for (j, face_vert) in face.iter_mut().enumerate() {
                if *face_vert == vertex_index as u32 {
                    // Duplicate the vertex in all faces except the first.
                    // The first face can just use the original index.
                    if i_f > 0 {
                        *face_vert = split_vertices.len() as u32;
                        split_vertices.push(split_vertices[vertex_index]);
                    }

                    i_face_vert = j;
                }
            }

            // Track any edges that have been duplicated.
            // The non sharp duplicated edges will be merged later.
            // Take advantage of every vertex being connected in a triangle.
            // Use the original vertices sharp edges refer to the original indices.
            let original_face = &vertex_indices[f * 3..f * 3 + 3];
            let mut e0 = [
                original_face[i_face_vert],
                original_face[(i_face_vert + 1) % 3],
            ];
            // Edges are undirected, so normalize the direction for each edge.
            e0.sort();
            duplicate_edges.insert(e0);

            let mut e1 = [
                original_face[i_face_vert],
                original_face[(i_face_vert + 2) % 3],
            ];
            e1.sort();
            duplicate_edges.insert(e1);
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
            split_edges(&[0.0, 1.0, 2.0], &[0, 1, 2], &[])
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
            split_edges(&[0.0, 1.0, 2.0, 3.0], &indices, &[[2, 3]])
        );
    }

    #[test]
    fn split_sharp_edges_two_quads() {
        // Two quads of two tris.
        // The topology shouldn't change.
        // 2 - 3 - 5
        // | \ | \ |
        // 0 - 1 - 4

        let indices = vec![0, 1, 2, 2, 1, 3, 3, 1, 4, 3, 4, 5];
        assert_eq!(
            (vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], indices.clone()),
            split_edges(
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                &indices,
                &[[2, 3], [3, 5], [0, 1], [1, 4]]
            )
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
            split_edges(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &indices, &[[1, 3]])
        );
    }

    #[test]
    fn split_sharp_edges_split_1_8cyli_dat() {
        // Example taken from p/1-8cyli.dat.
        // 3 - 0 - 4
        // | / | / |
        // 2 - 1 - 5

        // After splitting sharp edges.
        // 3 - 2 - 5
        // | / | / |
        // 0 - 1 - 4
        assert_eq!(
            (
                vec![2.0, 1.0, 0.0, 3.0, 5.0, 4.0],
                vec![0, 1, 2, 3, 0, 2, 1, 4, 5, 2, 1, 5]
            ),
            split_edges(
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                &[2, 1, 0, 3, 2, 0, 1, 5, 4, 0, 1, 4],
                &[[2, 1], [0, 3], [1, 5], [4, 0]]
            )
        );
    }
}

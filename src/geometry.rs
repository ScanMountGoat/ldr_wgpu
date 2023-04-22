use std::collections::{BTreeSet, HashMap};

use glam::Vec3;
use ldr_tools::LDrawColor;

use crate::normal::triangle_face_vertex_normals;

// TODO: Move this to ldr_tools and add tests.
#[derive(Clone)]
pub struct IndexedVertexData {
    pub vertices: Vec<crate::shader::model::VertexInput>,
    pub vertex_indices: Vec<u32>,
    pub edge_indices: Vec<u32>,
}

impl IndexedVertexData {
    pub fn from_geometry(geometry: &ldr_tools::LDrawGeometry) -> Self {
        let mut vertices = Vec::new();
        let mut vertex_indices = Vec::new();
        let mut edge_indices = Vec::new();
        // TODO: Edge colors?
        // TODO: Don't calculate grainy faces to save geometry?

        // TODO: missing color codes?
        // TODO: publicly expose color handling logic in ldr_tools.
        // TODO: handle the case where the face color list is empty?
        let mut vertex_cache = VertexCache::default();

        let (filtered_adjacent_faces, face_vertex_normals) =
            triangle_face_vertex_normals(&geometry.positions, &geometry.position_indices);

        for (i, vertex_index) in geometry.position_indices.iter().enumerate() {
            // Assume faces are already triangulated.
            // This means every 3 indices defines a new face.
            let face_index = i / 3;
            let face_color = geometry
                .face_colors
                .get(face_index)
                .unwrap_or(&geometry.face_colors[0]);

            // Each normal is uniquely determined by the set of filtered adjacent faces
            let adjacent_faces = filtered_adjacent_faces[i].clone();

            let vertex_position = geometry.positions[*vertex_index as usize];

            // Store separate indices for position and color.
            // TODO: Create a struct for this?
            // TODO: always ignore grainy slope information?
            // TODO: Pass this as a parameter?
            let face_vertex_key = VertexKey {
                position_index: *vertex_index,
                adjacent_faces,
                color: face_color.color,
            };

            let vertex_normal = face_vertex_normals[i];

            // Initially insert colors using the LDraw color code.
            // This will later be replaced by an RGBA color.
            // Take advantage of the fact that both use u32.
            let new_index = insert_vertex(
                face_vertex_key,
                vertex_position,
                vertex_normal,
                face_color.color,
                &mut vertex_cache,
                &mut vertices,
            );
            vertex_indices.push(new_index);
        }

        // The vertices have been reindexed, so use the mapping from earlier.
        // This ensures the sharp edge indices reference the correct vertices.
        // TODO: This color specific indexing probably won't work for normals.
        // TODO: Just create a separate vertex buffer without normals.
        edge_indices.extend(
            geometry
                .edge_position_indices
                .iter()
                .zip(geometry.is_edge_sharp.iter())
                .filter(|(_, sharp)| **sharp)
                .flat_map(|([v0, v1], _)| {
                    // Assume all black edges for now.
                    let i0 = insert_vertex(
                        VertexKey {
                            position_index: *v0,
                            adjacent_faces: BTreeSet::new(),
                            color: 0,
                        },
                        geometry.positions[*v0 as usize],
                        Vec3::ZERO,
                        0xFF000000,
                        &mut vertex_cache,
                        &mut vertices,
                    );
                    let i1 = insert_vertex(
                        VertexKey {
                            position_index: *v1,
                            adjacent_faces: BTreeSet::new(),
                            color: 0,
                        },
                        geometry.positions[*v1 as usize],
                        Vec3::ZERO,
                        0xFF000000,
                        &mut vertex_cache,
                        &mut vertices,
                    );

                    [i0, i1]
                }),
        );

        Self {
            vertices,
            vertex_indices,
            edge_indices,
        }
    }

    pub fn replace_colors(&mut self, current_color: u32, color_table: &HashMap<u32, LDrawColor>) {
        // Convert a color code to an RGBA color.
        for vertex in &mut self.vertices {
            vertex.color = rgba_color(vertex.color, current_color, color_table);
        }
    }
}

#[derive(Hash, PartialEq, Eq)]
struct VertexKey {
    position_index: u32,
    color: u32,
    // Normals are determined by the set of adjacent faces.
    // This helps define smoothing groups in the mesh.
    adjacent_faces: BTreeSet<usize>,
}

#[derive(Default)]
struct VertexCache {
    index_by_face_vertex: HashMap<VertexKey, u32>,
}

impl VertexCache {
    fn get(&self, key: &VertexKey) -> Option<u32> {
        self.index_by_face_vertex.get(key).copied()
    }

    fn insert(&mut self, k: VertexKey, v: u32) {
        self.index_by_face_vertex.insert(k, v);
    }

    fn len(&self) -> usize {
        self.index_by_face_vertex.len()
    }
}

// TODO: Simplify this and move to ldr_tools with tests?
fn insert_vertex(
    face_vertex_key: VertexKey,
    vertex_position: glam::Vec3,
    vertex_normal: glam::Vec3,
    vertex_color: u32,
    vertex_cache: &mut VertexCache,
    vertices: &mut Vec<crate::shader::model::VertexInput>,
) -> u32 {
    // A vertex is unique if its position and color are unique.
    // This allows attributes like color to be indexed by face.
    // Only the necessary vertices will be duplicated when reindexing.
    if let Some(cached_index) = vertex_cache.get(&face_vertex_key) {
        cached_index
    } else {
        let new_vertex = crate::shader::model::VertexInput {
            position: vertex_position,
            normal: vertex_normal.extend(0.0),
            color: vertex_color,
        };
        let new_index = vertex_cache.len() as u32;
        vertex_cache.insert(face_vertex_key, new_index);

        vertices.push(new_vertex);
        new_index
    }
}

fn rgba_color(color: u32, current_color: u32, color_table: &HashMap<u32, LDrawColor>) -> u32 {
    let replaced_color = if color == 16 { current_color } else { color };

    color_table
        .get(&replaced_color)
        .map(|c| {
            // TODO: What is the GPU endianness?
            u32::from_le_bytes(c.rgba_linear.map(|f| (f * 255.0) as u8))
        })
        .unwrap_or(0xFFFFFFFF)
}

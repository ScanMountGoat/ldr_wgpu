use glam::{vec3, Mat4};
use ldr_tools::ColorCode;
use log::info;
use normal::vertex_normals;
use std::ops::IndexMut;
use wgpu::util::DeviceExt;

pub const FOV_Y: f32 = 0.5;

pub const REQUIRED_FEATURES: wgpu::Features = wgpu::Features::EXPERIMENTAL_RAY_QUERY
    .union(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
    .union(wgpu::Features::TEXTURE_BINDING_ARRAY)
    .union(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

mod normal;
mod renderer;

pub use renderer::Renderer;

#[allow(dead_code)]
mod shader {
    pub mod blit {
        include!(concat!(env!("OUT_DIR"), "/blit.rs"));
    }
    pub mod shader {
        include!(concat!(env!("OUT_DIR"), "/shader.rs"));
    }
}

#[derive(Debug, Clone, Default)]
struct RawSceneComponents {
    vertices: Vec<shader::shader::Vertex>,
    indices: Vec<u32>,
    faces: Vec<shader::shader::Face>,
    geometries: Vec<SceneGeometry>,
    instances: Vec<SceneInstance>,
    images: Vec<image::RgbaImage>,
}

#[derive(Debug, Clone, Default)]
struct SceneGeometry {
    vertex_start_index: usize,
    vertex_count: usize,

    index_start_index: usize,
    index_count: usize,
}

#[derive(Debug, Clone, Default)]
struct SceneInstance {
    geometry_index: usize,
    transform: Mat4,
}

struct SceneComponents {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    faces: wgpu::Buffer,

    geometries: wgpu::Buffer,

    scene_instances: Vec<SceneInstance>,

    bottom_level_acceleration_structures: Vec<wgpu::Blas>,

    textures: Vec<wgpu::TextureView>,
}

fn upload_scene_components(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &RawSceneComponents,
) -> SceneComponents {
    let geometry_buffer_content = scene
        .geometries
        .iter()
        .map(|geometry| shader::shader::Geometry {
            vertex_start_index: geometry.vertex_start_index as u32,
            index_start_index: geometry.index_start_index as u32,
            _pad1: 0,
            _pad2: 0,
        })
        .collect::<Vec<_>>();

    let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertices"),
        contents: bytemuck::cast_slice(&scene.vertices),
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT,
    });
    let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Indices"),
        contents: bytemuck::cast_slice(&scene.indices),
        usage: wgpu::BufferUsages::INDEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT,
    });
    let faces = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Faces"),
        contents: bytemuck::cast_slice(&scene.faces),
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT,
    });
    let geometries = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Geometries"),
        contents: bytemuck::cast_slice(&geometry_buffer_content),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let textures = scene
        .images
        .iter()
        .map(|i| image_texture(device, queue, i))
        .collect();

    let (size_descriptors, bottom_level_acceleration_structures): (Vec<_>, Vec<_>) = scene
        .geometries
        .iter()
        .map(|geometry| {
            let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: geometry.vertex_count as u32,
                index_format: Some(wgpu::IndexFormat::Uint32),
                index_count: Some(geometry.index_count as u32),
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            };

            let blas = device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: None,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![size_desc.clone()],
                },
            );
            (size_desc, blas)
        })
        .unzip();

    let build_entries: Vec<_> = scene
        .geometries
        .iter()
        .zip(size_descriptors.iter())
        .zip(bottom_level_acceleration_structures.iter())
        .map(|((geometry, size), blas)| {
            let triangle_geometry = wgpu::BlasTriangleGeometry {
                size,
                vertex_buffer: &vertices,
                first_vertex: geometry.vertex_start_index as u32,
                vertex_stride: std::mem::size_of::<shader::shader::Vertex>() as u64,
                index_buffer: Some(&indices),
                first_index: Some(geometry.index_start_index as u32),
                transform_buffer: None,
                transform_buffer_offset: None,
            };

            wgpu::BlasBuildEntry {
                blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![triangle_geometry]),
            }
        })
        .collect();

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.build_acceleration_structures(build_entries.iter(), std::iter::empty());

    queue.submit(Some(encoder.finish()));

    SceneComponents {
        vertices,
        indices,
        geometries,
        faces,
        scene_instances: scene.instances.clone(),
        bottom_level_acceleration_structures,
        textures,
    }
}

fn load_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &str,
    ldraw_path: &str,
) -> SceneComponents {
    let start = std::time::Instant::now();

    // Weld vertices to take advantage of vertex caching/batching on the GPU.
    let settings = ldr_tools::GeometrySettings {
        triangulate: true,
        weld_vertices: true,
        stud_type: ldr_tools::StudType::Normal, // TODO: exceeds max buffer size for datsville with logo4
        ..Default::default()
    };
    let scene_instanced = ldr_tools::load_file_instanced(path, ldraw_path, &[], &settings);
    info!("Load LDraw file: {:?}", start.elapsed());

    let start = std::time::Instant::now();

    let scene = RawSceneComponents::new(scene_instanced);

    let components = upload_scene_components(device, queue, &scene);

    info!(
        "Process {} instances: {:?}",
        scene.instances.len(),
        start.elapsed()
    );

    components
}

impl RawSceneComponents {
    fn new(scene_instanced: ldr_tools::LDrawSceneInstanced) -> Self {
        let mut scene = Self::default();

        // TODO: should each geometry correspond to exactly one blas?
        // TODO: Process these in parallel?
        for ((name, color_code), transforms) in scene_instanced.geometry_world_transforms {
            let geometry = &scene_instanced.geometry_cache[&name];

            if let Some(info) = &geometry.texture_info {
                for png_bytes in &info.textures {
                    // TODO: The texture indices need to be remapped for per scene images.
                    let image =
                        image::load_from_memory_with_format(png_bytes, image::ImageFormat::Png)
                            .unwrap()
                            .into_rgba8();
                    scene.images.push(image);
                }
            }

            let start_vertex_index = scene.vertices.len();
            scene.add_vertices(geometry);

            scene.add_faces(color_code, geometry);

            let start_index_index = scene.indices.len();
            for i in &geometry.vertex_indices {
                scene.indices.push(*i);
            }

            let geometry_index = scene.geometries.len();
            scene.geometries.push(SceneGeometry {
                vertex_start_index: start_vertex_index,
                vertex_count: geometry.vertices.len(),
                index_start_index: start_index_index,
                index_count: geometry.vertex_indices.len(),
            });

            // TODO: Don't duplicate blas for same part with multiple colors?
            for transform in transforms {
                scene.instances.push(SceneInstance {
                    geometry_index,
                    transform,
                });
            }

            // TODO: images are per geometry but need to be per scene for ray tracing.
        }

        scene
    }

    fn add_vertices(&mut self, geometry: &ldr_tools::LDrawGeometry) {
        let normals = vertex_normals(&geometry.vertices, &geometry.vertex_indices);

        let uvs = geometry
            .texture_info
            .as_ref()
            .map(|info| info.uvs.as_slice())
            .unwrap_or_default();

        for (i, (v, n)) in geometry.vertices.iter().zip(&normals).enumerate() {
            // Hard surface normals work fine with lower precision.
            // This allows fitting vertices into a single vec4.
            let normal_unorm8 = (n * 0.5 + 0.5)
                .extend(0.0)
                .to_array()
                .map(|v| (v * 255.0) as u8);

            self.vertices.push(shader::shader::Vertex {
                pos: *v,
                normal: u32::from_le_bytes(normal_unorm8),
                uv: uvs
                    .get(i)
                    .copied()
                    .unwrap_or_default()
                    .extend(0.0)
                    .extend(0.0),
            });
        }
    }

    fn add_faces(&mut self, color_code: u32, geometry: &ldr_tools::LDrawGeometry) {
        let texture_indices = geometry
            .texture_info
            .as_ref()
            .map(|info| info.indices.as_slice())
            .unwrap_or_default();

        if geometry.face_colors.len() == 1 {
            for i in 0..geometry.vertex_indices.len() / 3 {
                let color = replace_color(geometry.face_colors[0], color_code);
                self.faces.push(shader::shader::Face {
                    color_code: color,
                    texture_index: texture_indices
                        .get(i)
                        .map(|u| (*u as i8) as i32)
                        .unwrap_or(-1),
                });
            }
        } else {
            for (i, color) in geometry.face_colors.iter().enumerate() {
                let color = replace_color(*color, color_code);
                self.faces.push(shader::shader::Face {
                    color_code: color,
                    texture_index: texture_indices
                        .get(i)
                        .map(|u| (*u as i8) as i32)
                        .unwrap_or(-1),
                });
            }
        }
    }
}
fn replace_color(color: ColorCode, current_color: ColorCode) -> ColorCode {
    // TODO: Make this part of ldr_tools
    if color == 16 {
        current_color
    } else {
        color
    }
}

pub struct Scene {
    bind_group1: shader::shader::bind_groups::BindGroup1,
}

impl Scene {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, path: &str, ldraw_path: &str) -> Self {
        let scene_components = load_scene(device, queue, path, ldraw_path);

        let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: None,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            max_instances: scene_components.scene_instances.len() as u32,
        });

        let mut tlas_package = wgpu::TlasPackage::new(tlas);

        for (i, instance) in scene_components.scene_instances.iter().enumerate() {
            let tlas_instance = tlas_package.index_mut(i);

            // TODO: Should each geometry correspond to exactly one blas?
            let blas_index = instance.geometry_index;

            let transform = instance.transform.transpose().to_cols_array()[..12]
                .try_into()
                .unwrap();
            *tlas_instance = Some(wgpu::TlasInstance::new(
                &scene_components.bottom_level_acceleration_structures[blas_index],
                transform,
                blas_index as u32,
                0xff,
            ));
        }

        // TODO: Build tlas and blas together?
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.build_acceleration_structures(std::iter::empty(), std::iter::once(&tlas_package));
        queue.submit(Some(encoder.finish()));

        let default_texture = default_black_texture(device, queue);

        let mut textures = [&default_texture; shader::shader::TEXTURE_COUNT as usize];
        for (t, texture) in textures.iter_mut().zip(&scene_components.textures) {
            *t = texture;
        }

        let bind_group1 = shader::shader::bind_groups::BindGroup1::from_bindings(
            device,
            shader::shader::bind_groups::BindGroupLayout1 {
                vertices: scene_components.vertices.as_entire_buffer_binding(),
                indices: scene_components.indices.as_entire_buffer_binding(),
                faces: scene_components.faces.as_entire_buffer_binding(),
                geometries: scene_components.geometries.as_entire_buffer_binding(),
                textures: &textures,
                acc_struct: tlas_package.tlas(),
            },
        );

        Scene { bind_group1 }
    }
}

pub fn calculate_camera_data(
    width: u32,
    height: u32,
    translation: glam::Vec3,
    rotation: glam::Vec3,
) -> shader::shader::Camera {
    let aspect = width as f32 / height as f32;

    // TODO: Why is this reversed vertically compared to normal?
    let view = glam::Mat4::from_translation(vec3(translation.x, -translation.y, translation.z))
        * glam::Mat4::from_rotation_x(-rotation.x)
        * glam::Mat4::from_rotation_y(rotation.y)
        * glam::Mat4::from_scale(vec3(-1.0, 1.0, 1.0));

    // TODO: Does this even matter for draw distance?
    let projection = glam::Mat4::perspective_rh(FOV_Y, aspect, 0.001, 10000.0);

    shader::shader::Camera {
        view: view,
        view_inv: view.inverse(),
        proj_inv: projection.inverse(),
    }
}

fn image_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &image::RgbaImage,
) -> wgpu::TextureView {
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("image"),
            size: wgpu::Extent3d {
                width: image.width(),
                height: image.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &image.as_raw(),
    );

    texture.create_view(&Default::default())
}

fn default_black_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("DEFAULT_TEXTURE"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &[0u8; 4 * 4 * 4],
    );

    texture.create_view(&Default::default())
}

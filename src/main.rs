use futures::executor::block_on;
use glam::{vec3, vec4, Mat4, Vec3};
use ldr_tools::LDrawSceneInstanced;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod shader;

struct InstancedMesh {
    mesh: Mesh,
    instances: Vec<MeshInstance>,
}

struct MeshInstance {
    world_transform: wgpu::Buffer,
    bind_group1: crate::shader::bind_groups::BindGroup1,
}

struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    config: wgpu::SurfaceConfiguration,

    translation: Vec3,
    rotation_xyz: Vec3,
    camera_buffer: wgpu::Buffer,

    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    // Render State
    bind_group0: crate::shader::bind_groups::BindGroup0,
    pipeline: wgpu::RenderPipeline,

    instanced_meshes: Vec<InstancedMesh>,
}

impl State {
    async fn new(window: &Window, scene: &LDrawSceneInstanced) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = unsafe { instance.create_surface(window).unwrap() };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::TEXTURE_COMPRESSION_BC,
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let size = window.inner_size();
        let surface_format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: Vec::new(),
        };
        surface.configure(&device, &config);

        let shader = crate::shader::create_shader_module(&device);
        let render_pipeline_layout = crate::shader::create_pipeline_layout(&device);

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[crate::shader::VertexInput::vertex_buffer_layout(
                    wgpu::VertexStepMode::Vertex,
                )],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(surface_format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let translation = vec3(0.0, -0.5, -20.0);
        let rotation_xyz = Vec3::ZERO;
        let mvp_matrix = mvp_matrix(size, translation, rotation_xyz);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::cast_slice(&[crate::shader::Camera { mvp_matrix }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group0 = crate::shader::bind_groups::BindGroup0::from_bindings(
            &device,
            crate::shader::bind_groups::BindGroupLayout0 {
                camera: camera_buffer.as_entire_buffer_binding(),
            },
        );

        let start = std::time::Instant::now();
        let instanced_meshes = load_instances(&device, scene);
        println!(
            "Load {:?} parts: {:?}",
            instanced_meshes.len(),
            start.elapsed()
        );

        let (depth_texture, depth_view) = create_depth_texture(&device, size);

        Self {
            surface,
            device,
            queue,
            size,
            config,
            pipeline,
            bind_group0,
            instanced_meshes,
            translation,
            rotation_xyz,
            camera_buffer,
            depth_texture,
            depth_view,
        }
    }

    fn update_camera(&self, size: winit::dpi::PhysicalSize<u32>) {
        let mvp_matrix = mvp_matrix(size, self.translation, self.rotation_xyz);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[mvp_matrix]));
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            let (depth_texture, depth_view) = create_depth_texture(&self.device, new_size);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let output_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });

        render_pass.set_pipeline(&self.pipeline);
        self.bind_group0.set(&mut render_pass);

        // Draw the instances of each unique part and color.
        // This allows reusing most of the rendering state for better performance.
        for instanced_mesh in &self.instanced_meshes {
            let mesh = &instanced_mesh.mesh;
            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            // Each instance has a unique transform.
            // TODO: Benchmark with actual instances instead.
            for instance in &instanced_mesh.instances {
                instance.bind_group1.set(&mut render_pass);
                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1)
            }
        }

        drop(render_pass);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Actually draw the frame.
        output.present();

        Ok(())
    }
}

fn create_depth_texture(
    device: &wgpu::Device,
    size: winit::dpi::PhysicalSize<u32>,
) -> (wgpu::Texture, wgpu::TextureView) {
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let depth_view = depth_texture.create_view(&Default::default());

    (depth_texture, depth_view)
}

fn load_instances(device: &wgpu::Device, scene: &LDrawSceneInstanced) -> Vec<InstancedMesh> {
    scene
        .geometry_world_transforms
        .iter()
        .map(|((name, _color), transforms)| {
            let geometry = &scene.geometry_cache[name];

            let mesh = create_mesh(device, geometry);

            let instances = transforms
                .iter()
                .map(|transform| {
                    let world_transform_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("world transform buffer"),
                            contents: bytemuck::cast_slice(&[*transform]),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                    let bind_group1 = crate::shader::bind_groups::BindGroup1::from_bindings(
                        device,
                        crate::shader::bind_groups::BindGroupLayout1 {
                            world_transform: world_transform_buffer.as_entire_buffer_binding(),
                        },
                    );

                    MeshInstance {
                        world_transform: world_transform_buffer,
                        bind_group1,
                    }
                })
                .collect();

            InstancedMesh { mesh, instances }
        })
        .collect()
}

fn create_mesh(device: &wgpu::Device, geometry: &ldr_tools::LDrawGeometry) -> Mesh {
    // TODO: wgsl_to_wgpu should support non strict offset checking for vertex buffer structs.
    // This allows desktop applications to use vec3 for positions.
    let positions: Vec<_> = geometry
        .vertices
        .iter()
        .map(|v| vec4(v.x, v.y, v.z, 1.0))
        .collect();
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vertex buffer"),
        contents: bytemuck::cast_slice(&positions),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // TODO: triangulate.
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vertex buffer"),
        contents: bytemuck::cast_slice(&geometry.vertex_indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    Mesh {
        vertex_buffer,
        index_buffer,
        index_count: geometry.vertex_indices.len() as u32,
    }
}

fn mvp_matrix(
    size: winit::dpi::PhysicalSize<u32>,
    translation: glam::Vec3,
    rotation: glam::Vec3,
) -> glam::Mat4 {
    let aspect = size.width as f32 / size.height as f32;

    // wgpu and LDraw have different coordinate systems.
    let axis_correction = Mat4::from_rotation_x(180.0f32.to_radians());

    let model_view_matrix = glam::Mat4::from_translation(translation)
        * glam::Mat4::from_rotation_x(rotation.x)
        * glam::Mat4::from_rotation_y(rotation.y)
        * axis_correction;
    let perspective_matrix = glam::Mat4::perspective_rh(0.5, aspect, 0.1, 1000.0);

    perspective_matrix * model_view_matrix
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let ldraw_path = &args[1];
    let path = &args[2];

    let start = std::time::Instant::now();
    let scene = ldr_tools::load_file_instanced(path, ldraw_path);
    println!("Load scene: {:?}", start.elapsed());

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(concat!("ldr_wgpu ", env!("CARGO_PKG_VERSION")))
        .build(&event_loop)
        .unwrap();

    let mut state = block_on(State::new(&window, &scene));
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(physical_size) => {
                state.resize(*physical_size);
                state.update_camera(*physical_size);
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                state.resize(**new_inner_size);
            }
            WindowEvent::KeyboardInput { input, .. } => {
                // Basic camera controls using arrow keys.
                let size = window.inner_size();
                if let Some(keycode) = input.virtual_keycode {
                    match keycode {
                        VirtualKeyCode::Left => state.rotation_xyz.y -= 0.1,
                        VirtualKeyCode::Right => state.rotation_xyz.y += 0.1,
                        VirtualKeyCode::Up => state.translation.z += 0.5,
                        VirtualKeyCode::Down => state.translation.z -= 0.5,
                        _ => (),
                    }
                }
                state.update_camera(size);
            }
            _ => {}
        },
        Event::RedrawRequested(_) => match state.render() {
            Ok(_) => {}
            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
            Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
            Err(e) => eprintln!("{e:?}"),
        },
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}

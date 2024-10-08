use futures::executor::block_on;
use ldr_tools::{GeometrySettings, StudType};
use log::{debug, error, info};
use winit::{
    event::*,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    required_features: wgpu::Features,
    config: wgpu::SurfaceConfiguration,
}

impl<'a> State<'a> {
    async fn new(window: &'a Window, format: wgpu::TextureFormat) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        debug!("{:#?}", adapter.get_info());

        let supported_features = adapter.features();
        let required_features = ldr_wgpu::required_features(supported_features);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .unwrap();

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: Vec::new(),
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
            required_features,
        }
    }
}

fn main() {
    // Ignore most wgpu logs to avoid flooding the console.
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("ldr_wgpu", log::LevelFilter::Info)
        .init()
        .unwrap();

    let args: Vec<_> = std::env::args().collect();
    let ldraw_path = &args[1];
    let path = &args[2];

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title(concat!("ldr_wgpu ", env!("CARGO_PKG_VERSION")))
        .build(&event_loop)
        .unwrap();

    // Choose a format that's guaranteed to be supported.
    let format = wgpu::TextureFormat::Bgra8UnormSrgb;

    let mut state = block_on(State::new(&window, format));

    let size = window.inner_size();
    let mut renderer = ldr_wgpu::Renderer::new(
        &state.device,
        size.width,
        size.height,
        format,
        state.required_features,
    );

    // Weld vertices to take advantage of vertex caching/batching on the GPU.
    let start = std::time::Instant::now();
    let settings = GeometrySettings {
        triangulate: true,
        weld_vertices: true,
        stud_type: StudType::HighContrast,
        ..Default::default()
    };
    let scene = ldr_tools::load_file_instanced(path, ldraw_path, &[], &settings);
    info!("Load scene: {:?}", start.elapsed());

    let color_table = ldr_tools::load_color_table(ldraw_path);

    let mut render_data = ldr_wgpu::RenderData::new(&state.device, &scene, &color_table);

    event_loop
        .run(|event, target| match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(size) => {
                    state.config.width = size.width;
                    state.config.height = size.height;
                    state.surface.configure(&state.device, &state.config);

                    renderer.resize(&state.device, size.width, size.height, format);
                    renderer.update_camera(&state.queue, size.width, size.height);
                    window.request_redraw();
                }
                WindowEvent::ScaleFactorChanged { .. } => {}
                WindowEvent::RedrawRequested => {
                    match state.surface.get_current_texture() {
                        Ok(output) => {
                            let output_view = output
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());

                            renderer.render(
                                &state.device,
                                &state.queue,
                                &mut render_data,
                                &output_view,
                            );
                            output.present();
                        }
                        Err(wgpu::SurfaceError::Lost) => {
                            let size = window.inner_size();
                            renderer.resize(&state.device, size.width, size.height, format)
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                        Err(e) => error!("{e:?}"),
                    }
                    window.request_redraw();
                }
                _ => {
                    let size = window.inner_size();
                    renderer.handle_input(event, size);
                    renderer.update_camera(&state.queue, size.width, size.height);
                    window.request_redraw();
                }
            },
            _ => (),
        })
        .unwrap();
}

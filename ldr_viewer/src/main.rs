use futures::executor::block_on;
use ldr_tools::{GeometrySettings, StudType};
use log::{error, info};
use winit::{event::*, event_loop::EventLoop, window::WindowBuilder};

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

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title(concat!("ldr_wgpu ", env!("CARGO_PKG_VERSION")))
        .build(&event_loop)
        .unwrap();

    let mut state = block_on(ldr_wgpu::State::new(&window, &scene, &color_table));
    event_loop
        .run(|event, target| match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                    state.update_camera(*physical_size);
                    window.request_redraw();
                }
                WindowEvent::ScaleFactorChanged { .. } => {}
                WindowEvent::RedrawRequested => {
                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                        Err(e) => error!("{e:?}"),
                    }
                    window.request_redraw();
                }
                _ => {
                    state.handle_input(event);
                    state.update_camera(window.inner_size());
                    window.request_redraw();
                }
            },
            _ => (),
        })
        .unwrap();
}

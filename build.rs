use wgsl_to_wgpu::{create_shader_module_embedded, MatrixVectorTypes, WriteOptions};

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    write_shader("src/shader/model.wgsl", format!("{out_dir}/model.rs"));
    write_shader("src/shader/culling.wgsl", format!("{out_dir}/culling.rs"));
    write_shader(
        "src/shader/depth_pyramid.wgsl",
        format!("{out_dir}/depth_pyramid.rs"),
    );
    write_shader(
        "src/shader/blit_depth.wgsl",
        format!("{out_dir}/blit_depth.rs"),
    );
    write_shader(
        "src/shader/visibility.wgsl",
        format!("{out_dir}/visibility.rs"),
    );
    write_shader("src/shader/scan.wgsl", format!("{out_dir}/scan.rs"));
    write_shader("src/shader/scan_add.wgsl", format!("{out_dir}/scan_add.rs"));
}

fn write_shader(wgsl_path: &str, output_path: String) {
    println!("cargo:rerun-if-changed={wgsl_path}");

    let wgsl_source = std::fs::read_to_string(wgsl_path).unwrap();

    // Generate the Rust bindings and write to a file.
    let text = create_shader_module_embedded(
        &wgsl_source,
        WriteOptions {
            derive_bytemuck_vertex: true,
            derive_bytemuck_host_shareable: true,
            matrix_vector_types: MatrixVectorTypes::Glam,
            ..Default::default()
        },
    )
    .unwrap();

    std::fs::write(output_path, text.as_bytes()).unwrap();
}

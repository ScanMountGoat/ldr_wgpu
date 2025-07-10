use wgsl_to_wgpu::{create_shader_modules, demangle_identity, MatrixVectorTypes, WriteOptions};

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    write_shader("src/shader/shader.wgsl", format!("{out_dir}/shader.rs"));
    write_shader("src/shader/blit.wgsl", format!("{out_dir}/blit.rs"));
}

fn write_shader(wgsl_path: &str, output_path: String) {
    println!("cargo:rerun-if-changed={wgsl_path}");

    let wgsl_source = std::fs::read_to_string(wgsl_path).unwrap();

    // Generate the Rust bindings and write to a file.
    let text = create_shader_modules(
        &wgsl_source,
        WriteOptions {
            derive_bytemuck_vertex: true,
            derive_bytemuck_host_shareable: true,
            matrix_vector_types: MatrixVectorTypes::Glam,
            ..Default::default()
        },
        demangle_identity,
    )
    .unwrap();

    std::fs::write(output_path, text.as_bytes()).unwrap();
}

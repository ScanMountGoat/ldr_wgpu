#[allow(dead_code)]
pub mod blit_depth {
    include!(concat!(env!("OUT_DIR"), "/blit_depth.rs"));
}
#[allow(dead_code)]
pub mod culling {
    include!(concat!(env!("OUT_DIR"), "/culling.rs"));
}
#[allow(dead_code)]
pub mod depth_pyramid {
    include!(concat!(env!("OUT_DIR"), "/depth_pyramid.rs"));
}
#[allow(dead_code)]
pub mod model {
    include!(concat!(env!("OUT_DIR"), "/model.rs"));
}
#[allow(dead_code)]
pub mod scan {
    include!(concat!(env!("OUT_DIR"), "/scan.rs"));
}
#[allow(dead_code)]
pub mod scan_add {
    include!(concat!(env!("OUT_DIR"), "/scan_add.rs"));
}
#[allow(dead_code)]
pub mod visibility {
    include!(concat!(env!("OUT_DIR"), "/visibility.rs"));
}

struct Uniforms {
    [[size(80)]] _padding: u32;
    depth_add: f32;
    depth_mul: f32;
};

[[group(0), binding(0)]]
var<uniform> uniforms: Uniforms;
[[group(0), binding(1)]]
var r_output_texture: texture_storage_2d<r8unorm, write>;
[[group(0), binding(2)]]
var r_ssao_texture: texture_2d<f32>;
[[group(0), binding(3)]]
var r_depth_texture: texture_2d<f32>;

let KERNEL_RADIUS: i32 = 5;

fn screen_space_depth_to_view_space_z(d: f32) -> f32 {
    return uniforms.depth_mul / (uniforms.depth_add - d);
}

fn get_view_space_z(p: vec2<i32>) -> f32 {
    let d = textureLoad(r_depth_texture, p, 0).x;
    return screen_space_depth_to_view_space_z(d);
}

fn cross_bilateral_weight(r: f32, z: f32, z0: f32) -> f32 {
    let blur_sigma = (f32(KERNEL_RADIUS) + 1.0) * 0.5;
    let blur_falloff = 1.0 / (2.0 * blur_sigma * blur_sigma);
    
    // assuming that d and d0 are pre-scaled linear depths 
    let dz = z0 - z;
    return exp2(-r*r*blur_falloff - dz*dz);
}

fn process_sample(
    p: vec2<i32>,
    r: f32,
    z0: f32,
    total_ao: ptr<function, f32>,
    total_weight: ptr<function, f32>
) {
    let z = get_view_space_z(p);
    let ao = textureLoad(r_ssao_texture, p, 0).x;

    let weight = cross_bilateral_weight(r, z, z0);
    *total_ao = *total_ao + weight * ao;
    *total_weight = *total_weight + weight;
}

[[stage(compute), workgroup_size(8, 8)]]
fn blur_x([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    let size = textureDimensions(r_output_texture);
    let p0 = vec2<i32>(global_invocation_id.xy);

    if (!all(p0 < size)) {
        return;
    }

    let z0: f32 = get_view_space_z(p0);
    var total_ao: f32 = 0.0;
    var total_weight: f32 = 0.0;

    let x1 = max(p0.x - KERNEL_RADIUS, 0);
    let x2 = min(p0.x + KERNEL_RADIUS, size.x - 1);

    for (var x = x1; x <= x2; x = x + 1) {
        let p = vec2<i32>(x, p0.y);
        let r = abs(f32(p0.x - x));
        process_sample(p, r, z0, &total_ao, &total_weight);
    }

    let ao = total_ao / total_weight;
    textureStore(r_output_texture, p0, vec4<f32>(ao, ao, ao, ao));
}

[[stage(compute), workgroup_size(8, 8)]]
fn blur_y([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    let size = textureDimensions(r_output_texture);
    let p0 = vec2<i32>(global_invocation_id.xy);

    if (!all(p0 < size)) {
        return;
    }

    let z0: f32 = get_view_space_z(p0);
    var total_ao: f32 = 0.0;
    var total_weight: f32 = 0.0;

    let y1 = max(p0.y - KERNEL_RADIUS, 0);
    let y2 = min(p0.y + KERNEL_RADIUS, size.y - 1);

    for (var y = y1; y <= y2; y = y + 1) {
        let p = vec2<i32>(p0.x, y);
        let r = abs(f32(p0.y - y));
        process_sample(p, r, z0, &total_ao, &total_weight);
    }

    let ao = total_ao / total_weight;
    textureStore(r_output_texture, p0, vec4<f32>(ao, ao, ao, ao));
}

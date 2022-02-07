struct Uniforms {
    projection: mat4x4<f32>;
    uv_to_view_space_add: vec2<f32>;
    uv_to_view_space_mul: vec2<f32>;
    depth_add: f32;
    depth_mul: f32;
    noise_offset: vec2<u32>;
    sample_count: u32;
    radius: f32;
    bias: f32;
};

struct Samples {
    data: array<vec4<f32>>;
};

[[group(0), binding(0)]]
var<uniform> uniforms: Uniforms;
[[group(0), binding(1)]]
var<storage, read> samples: Samples;
[[group(0), binding(2)]]
var r_output_texture: texture_storage_2d<r8unorm, write>;
[[group(0), binding(3)]]
var r_depth_texture: texture_2d<f32>;
[[group(0), binding(4)]]
var r_normal_texture: texture_2d<f32>;
[[group(0), binding(5)]]
var r_noise_texture: texture_2d<f32>;

fn screen_space_depth_to_view_space_z(d: f32) -> f32 {
    return uniforms.depth_mul / (uniforms.depth_add - d);
}

fn uv_to_view_space_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let z = screen_space_depth_to_view_space_z(depth);
    return vec3<f32>((uv * uniforms.uv_to_view_space_mul + uniforms.uv_to_view_space_add) * -z, z);
}

fn load_random_vec(frag_coord: vec2<u32>) -> vec3<f32> {
    let p = (frag_coord + uniforms.noise_offset) % vec2<u32>(textureDimensions(r_noise_texture));
    return vec3<f32>(textureLoad(r_noise_texture, vec2<i32>(p), 0).xy, 0.0);
}

[[stage(compute), workgroup_size(8, 8)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    let size = vec2<u32>(textureDimensions(r_output_texture));
    let frag_coord = global_invocation_id.xy;

    if (!all(frag_coord < size)) {
        return;
    }

    let depth = textureLoad(r_depth_texture, vec2<i32>(frag_coord), 0).x;
    if (depth == 1.0) {
        textureStore(r_output_texture, vec2<i32>(frag_coord), vec4<f32>(1.0, 1.0, 1.0, 1.0));
        return;
    }

    let frag_uv = (vec2<f32>(frag_coord) + 0.5) / vec2<f32>(size);
    let view_pos = uv_to_view_space_position(frag_uv, depth);
    let normal = textureLoad(r_normal_texture, vec2<i32>(frag_coord), 0).xyz;
    let random_vec = load_random_vec(frag_coord);

    let tangent = normalize(random_vec - normal * dot(random_vec, normal));
    let bitangent = cross(normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, normal);

    var occlusion = 0.0;
    for (var i = 0u; i < uniforms.sample_count; i = i + 1u) {
        let sample_vec = uniforms.radius * (tbn * samples.data[i].xyz);
        let sample_pos = view_pos + sample_vec;

        let sample_clip_pos = uniforms.projection * vec4<f32>(sample_pos, 1.0);
        let sample_ndc = sample_clip_pos.xy / sample_clip_pos.w;
        let sample_uv = sample_ndc * vec2<f32>(0.5, -0.5) + 0.5;
        var sample_coords = vec2<i32>(floor(sample_uv * vec2<f32>(size)));
        sample_coords = clamp(sample_coords, vec2<i32>(0), vec2<i32>(size) - 1);

        let sample_depth = textureLoad(r_depth_texture, sample_coords, 0).x;
        if (sample_depth == 1.0) {
            continue;
        }

        let z = screen_space_depth_to_view_space_z(sample_depth);
        let range_check = smoothStep(0.0, 1.0, uniforms.radius / abs(view_pos.z - z));
        occlusion = occlusion + select(0.0, 1.0, z >= sample_pos.z + uniforms.bias) * range_check;
    }

    occlusion = 1.0 - (occlusion / f32(uniforms.sample_count));
    textureStore(
        r_output_texture,
        vec2<i32>(frag_coord),
        vec4<f32>(occlusion, occlusion, occlusion, occlusion)
    );
}
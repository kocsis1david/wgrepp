struct FsUniforms {
    ssao_enabled: u32;
};

[[group(0), binding(0)]]
var<uniform> r_locals: FsUniforms;

[[stage(vertex)]]
fn vs_main([[builtin(vertex_index)]] vertex_index: u32) -> [[builtin(position)]] vec4<f32> {
    // TODO: find a better way to handle conditions
    if (vertex_index == 0u) {
        return vec4<f32>(-1.0, -1.0, 0.0, 1.0);
    } else {
        if (vertex_index == 1u) {
            return vec4<f32>(3.0, -1.0, 0.0, 1.0);
        } else {
            return vec4<f32>(-1.0, 3.0, 0.0, 1.0);
        }
    }
}

[[group(0), binding(1)]]
var r_color_texture: texture_storage_2d<rgba16float, read>;
[[group(0), binding(2)]]
var r_ssao_texture: texture_storage_2d<r8unorm, read>;

[[stage(fragment)]]
fn fs_main([[builtin(position)]] position: vec4<f32>) -> [[location(0)]] vec4<f32> {
    let p = vec2<i32>(position.xy);
    let color = textureLoad(r_color_texture, p);

    if (r_locals.ssao_enabled == 1u) {
        let ssao_value = textureLoad(r_ssao_texture, p).r;
        let ssao_scale = color.a;

        return vec4<f32>(color.rgb * mix(1.0, ssao_value, ssao_scale), 1.0);
    } else {
        return vec4<f32>(color.rgb, 1.0);
    }
}

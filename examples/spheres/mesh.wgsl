struct TemporalMat4x4 {
    previous: mat4x4<f32>;
    current: mat4x4<f32>;
};

[[block]]
struct VsUniforms {
    view_proj_matrix: TemporalMat4x4;
};
[[group(0), binding(0)]]
var<uniform> r_locals: VsUniforms;

struct Instance {
    model_matrix: TemporalMat4x4;
    normal_matrix: mat4x4;
    color: vec4<f32>;
};

[[block]]
struct VsInstances {
    instances: array<Instance>;
};

[[group(0), binding(1)]]
var<storage, read> r_instances: VsInstances;

struct VsOutput {
    [[location(0)]] normal: vec3<f32>;
    [[location(1)]] color: vec4<f32>;
    [[builtin(position)]] position: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] position: vec3<f32>,
    [[location(1)]] normal: vec3<f32>,
    [[builtin(instance_index)]] instance_index: u32,
) -> VsOutput {
    let instance = r_instances.instances[instance_index];

    var out: VsOutput;
    out.normal = (r_locals.normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    out.color = instance.color;
    out.position = r_locals.view_proj_matrix.current
        * instance.model_matrix.current * vec4<f32>(position, 1.0);
    return out;
}

struct FsOutput {
    [[location(0)]] color: vec4<f32>;
    [[location(1)]] normal: vec2<f32>;
};

[[stage(fragment)]]
fn fs_main(in: VsOutput) -> FsOutput {
    let light_dir = normalize(vec3<f32>(-0.5, 1.0, 1.0));
    let normal = normalize(in.normal);
    var x = clamp(dot(light_dir, normal), 0.0, 1.0) * 0.75 + 0.25;

    var out: FsOutput;
    out.color = vec4<f32>(in.color.rgb * x, in.color.a);
    out.normal = normal.xy;
    return out;
}

struct VsUniforms {
    view: mat4x4<f32>;
    proj: mat4x4<f32>;
};

[[group(0), binding(0)]]
var<uniform> r_locals: VsUniforms;

struct Instance {
    model: mat4x4<f32>;
    color: vec4<f32>;
};

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
    out.normal = (r_locals.view * vec4<f32>(normal, 0.0)).xyz;
    out.color = instance.color;
    out.position = r_locals.proj * r_locals.view * instance.model * vec4<f32>(position, 1.0);
    return out;
}

struct FsOutput {
    [[location(0)]] color: vec4<f32>;
    [[location(1)]] normal: vec4<f32>;
};

[[stage(fragment)]]
fn fs_main(in: VsOutput) -> FsOutput {
    let sun_dir = normalize(vec3<f32>(-0.5, 1.0, 1.0));
    let normal = normalize(in.normal);
    let ambient = 0.25;
    let sun = clamp(dot(sun_dir, normal), 0.0, 1.0) * 0.75;
    let total_light = sun + ambient;

    var out: FsOutput;
    out.color = vec4<f32>(in.color.rgb * total_light, ambient / total_light);
    out.normal = vec4<f32>(normal, 0.0);
    return out;
}

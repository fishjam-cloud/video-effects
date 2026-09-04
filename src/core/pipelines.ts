/// <reference types="@webgpu/types" preserve="true" />

import { BUFFER_USAGE, TEXTURE_USAGE } from "./constants";

const FRAGMENT_STAGE = 0x02;

const FULL_SCREEN_VERTEX = /* wgsl */ `
struct VertexOutput { @builtin(position) position: vec4f, @location(0) uv: vec2f };
@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOutput {
  var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let position = positions[index];
  return VertexOutput(vec4f(position, 0.0, 1.0), vec2f((position.x + 1.0) * 0.5, 1.0 - (position.y + 1.0) * 0.5));
}
`;

const COPY_FRAGMENT = /* wgsl */ `
@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;
@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  return textureSample(sourceTexture, sourceSampler, input.uv);
}
`;

const BLUR_FRAGMENT = /* wgsl */ `
@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;
@group(0) @binding(2) var<uniform> blurParams: vec4f;
@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let dimensions = vec2f(textureDimensions(sourceTexture));
  let step = blurParams.xy * blurParams.z / dimensions;
  var color = textureSample(sourceTexture, sourceSampler, input.uv) * 0.227027;
  color += textureSample(sourceTexture, sourceSampler, input.uv + step) * 0.1945946;
  color += textureSample(sourceTexture, sourceSampler, input.uv - step) * 0.1945946;
  color += textureSample(sourceTexture, sourceSampler, input.uv + step * 2.0) * 0.1216216;
  color += textureSample(sourceTexture, sourceSampler, input.uv - step * 2.0) * 0.1216216;
  color += textureSample(sourceTexture, sourceSampler, input.uv + step * 3.0) * 0.054054;
  color += textureSample(sourceTexture, sourceSampler, input.uv - step * 3.0) * 0.054054;
  color += textureSample(sourceTexture, sourceSampler, input.uv + step * 4.0) * 0.016216;
  color += textureSample(sourceTexture, sourceSampler, input.uv - step * 4.0) * 0.016216;
  return color;
}
`;

const MASK_COMPOSITE_FRAGMENT = /* wgsl */ `
struct CompositeParams {
  maskRow0: vec4f,
  maskRow1: vec4f,
  edgeFeather: f32,
  _padding: vec3f,
};
@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;
@group(1) @binding(0) var backgroundTexture: texture_2d<f32>;
@group(1) @binding(1) var backgroundSampler: sampler;
@group(2) @binding(0) var maskTexture: texture_2d<f32>;
@group(2) @binding(1) var maskSampler: sampler;
@group(3) @binding(0) var<uniform> params: CompositeParams;
@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let maskUv = vec2f(
    dot(params.maskRow0.xyz, vec3f(input.uv, 1.0)),
    dot(params.maskRow1.xyz, vec3f(input.uv, 1.0)),
  );
  let confidence = textureSample(maskTexture, maskSampler, maskUv).r;
  let alpha = smoothstep(0.5 - params.edgeFeather, 0.5 + params.edgeFeather, confidence);
  return mix(textureSample(backgroundTexture, backgroundSampler, input.uv), textureSample(sourceTexture, sourceSampler, input.uv), alpha);
}
`;

const IMAGE_COMPOSITE_FRAGMENT = /* wgsl */ `
struct ImageParams {
  maskRow0: vec4f,
  maskRow1: vec4f,
  imageScale: vec2f,
  imageOffset: vec2f,
  edgeFeather: f32,
  contains: f32,
  _padding: vec2f,
  backgroundColor: vec4f,
};
@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;
@group(1) @binding(0) var imageTexture: texture_2d<f32>;
@group(1) @binding(1) var imageSampler: sampler;
@group(2) @binding(0) var maskTexture: texture_2d<f32>;
@group(2) @binding(1) var maskSampler: sampler;
@group(3) @binding(0) var<uniform> params: ImageParams;
@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let maskUv = vec2f(
    dot(params.maskRow0.xyz, vec3f(input.uv, 1.0)),
    dot(params.maskRow1.xyz, vec3f(input.uv, 1.0)),
  );
  let imageUv = input.uv * params.imageScale + params.imageOffset;
  let insideImage = all(imageUv >= vec2f(0.0)) && all(imageUv <= vec2f(1.0));
  let imageColor = select(params.backgroundColor, textureSample(imageTexture, imageSampler, imageUv), insideImage || params.contains < 0.5);
  let confidence = textureSample(maskTexture, maskSampler, maskUv).r;
  let alpha = smoothstep(0.5 - params.edgeFeather, 0.5 + params.edgeFeather, confidence);
  return mix(imageColor, textureSample(sourceTexture, sourceSampler, input.uv), alpha);
}
`;

function textureLayout(device: GPUDevice): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: FRAGMENT_STAGE,
        texture: { sampleType: "float" },
      },
      {
        binding: 1,
        visibility: FRAGMENT_STAGE,
        sampler: { type: "filtering" },
      },
    ],
  });
}

function uniformLayout(device: GPUDevice): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: FRAGMENT_STAGE, buffer: { type: "uniform" } },
    ],
  });
}

function createPipeline(
  device: GPUDevice,
  code: string,
  outputFormat: GPUTextureFormat,
  layouts: readonly GPUBindGroupLayout[],
  label: string,
): GPURenderPipeline {
  const module = device.createShaderModule({
    code: `${FULL_SCREEN_VERTEX}\n${code}`,
    label,
  });
  return device.createRenderPipeline({
    label,
    layout: device.createPipelineLayout({ bindGroupLayouts: [...layouts] }),
    vertex: { module, entryPoint: "vertexMain" },
    fragment: {
      module,
      entryPoint: "fragmentMain",
      targets: [{ format: outputFormat }],
    },
    primitive: { topology: "triangle-list" },
  });
}

export interface BackgroundPipelines {
  readonly sampler: GPUSampler;
  readonly textureLayout: GPUBindGroupLayout;
  readonly uniformLayout: GPUBindGroupLayout;
  readonly copy: GPURenderPipeline;
  readonly blur: GPURenderPipeline;
  readonly maskComposite: GPURenderPipeline;
  readonly imageComposite: GPURenderPipeline;
}

export function createBackgroundPipelines(
  device: GPUDevice,
  outputFormat: GPUTextureFormat,
): BackgroundPipelines {
  const textures = textureLayout(device);
  const uniforms = uniformLayout(device);
  return {
    sampler: device.createSampler({
      minFilter: "linear",
      magFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    }),
    textureLayout: textures,
    uniformLayout: uniforms,
    copy: createPipeline(
      device,
      COPY_FRAGMENT,
      outputFormat,
      [textures],
      "fishjam-video-effect-copy",
    ),
    blur: createPipeline(
      device,
      BLUR_FRAGMENT,
      "rgba8unorm",
      [textures, uniforms],
      "fishjam-video-effect-blur",
    ),
    maskComposite: createPipeline(
      device,
      MASK_COMPOSITE_FRAGMENT,
      outputFormat,
      [textures, textures, textures, uniforms],
      "fishjam-video-effect-mask-composite",
    ),
    imageComposite: createPipeline(
      device,
      IMAGE_COMPOSITE_FRAGMENT,
      outputFormat,
      [textures, textures, textures, uniforms],
      "fishjam-video-effect-image-composite",
    ),
  };
}

export function createUniformBuffer(
  device: GPUDevice,
  size: number,
  label: string,
): GPUBuffer {
  return device.createBuffer({
    label,
    size,
    usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
  });
}

export function createSampleTexture(
  device: GPUDevice,
  width: number,
  height: number,
  label: string,
): { texture: GPUTexture; view: GPUTextureView } {
  const texture = device.createTexture({
    label,
    format: "rgba8unorm",
    size: [width, height],
    usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.RENDER_ATTACHMENT,
  });
  return { texture, view: texture.createView() };
}

export function createTextureBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  view: GPUTextureView,
  sampler: GPUSampler,
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: view },
      { binding: 1, resource: sampler },
    ],
  });
}

export function createUniformBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  buffer: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [{ binding: 0, resource: { buffer } }],
  });
}

export function drawFullscreen(
  encoder: GPUCommandEncoder,
  target: GPUTextureView,
  pipeline: GPURenderPipeline,
  groups: readonly GPUBindGroup[],
): void {
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: target,
        clearValue: [0, 0, 0, 1],
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  pass.setPipeline(pipeline);
  groups.forEach((group, index) => pass.setBindGroup(index, group));
  pass.draw(3);
  pass.end();
}

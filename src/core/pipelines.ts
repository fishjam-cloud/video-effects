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

// Dual Kawase blur (Bjørge, SIGGRAPH 2015): halve the resolution a few times with five bilinear
// taps per pass, then double it back with eight. Wide and smooth for a handful of cheap passes,
// with no uniforms: every tap offset comes from the source texture's own size.
const DOWNSAMPLE_FRAGMENT = /* wgsl */ `
@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;
@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let offset = 1.0 / vec2f(textureDimensions(sourceTexture));
  var color = textureSample(sourceTexture, sourceSampler, input.uv) * 4.0;
  color += textureSample(sourceTexture, sourceSampler, input.uv - offset);
  color += textureSample(sourceTexture, sourceSampler, input.uv + offset);
  color += textureSample(sourceTexture, sourceSampler, input.uv + vec2f(offset.x, -offset.y));
  color += textureSample(sourceTexture, sourceSampler, input.uv - vec2f(offset.x, -offset.y));
  return color / 8.0;
}
`;

const UPSAMPLE_FRAGMENT = /* wgsl */ `
@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;
@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let offset = 0.5 / vec2f(textureDimensions(sourceTexture));
  var color = textureSample(sourceTexture, sourceSampler, input.uv + vec2f(-offset.x * 2.0, 0.0));
  color += textureSample(sourceTexture, sourceSampler, input.uv + vec2f(-offset.x, offset.y)) * 2.0;
  color += textureSample(sourceTexture, sourceSampler, input.uv + vec2f(0.0, offset.y * 2.0));
  color += textureSample(sourceTexture, sourceSampler, input.uv + vec2f(offset.x, offset.y)) * 2.0;
  color += textureSample(sourceTexture, sourceSampler, input.uv + vec2f(offset.x * 2.0, 0.0));
  color += textureSample(sourceTexture, sourceSampler, input.uv + vec2f(offset.x, -offset.y)) * 2.0;
  color += textureSample(sourceTexture, sourceSampler, input.uv + vec2f(0.0, -offset.y * 2.0));
  color += textureSample(sourceTexture, sourceSampler, input.uv + vec2f(-offset.x, -offset.y)) * 2.0;
  return color / 12.0;
}
`;

// Shared by every shader that reads the person mask. `erode` shrinks the person area by that
// many mask texels (as uv) so the sharp region hugs the body instead of spilling around it, and
// `threshold`/`feather` turn the eroded confidence into a soft blend weight. Outside the square the
// model looked at the mask would clamp to its edge and smear the border across the strip, so those
// pixels count as background.
const PERSON_ALPHA_FUNCTION = /* wgsl */ `
fn erodedMask(maskUv: vec2f, erode: vec2f) -> f32 {
  var confidence = textureSample(maskTexture, maskSampler, maskUv).r;
  confidence = min(confidence, textureSample(maskTexture, maskSampler, maskUv + vec2f(erode.x, 0.0)).r);
  confidence = min(confidence, textureSample(maskTexture, maskSampler, maskUv - vec2f(erode.x, 0.0)).r);
  confidence = min(confidence, textureSample(maskTexture, maskSampler, maskUv + vec2f(0.0, erode.y)).r);
  confidence = min(confidence, textureSample(maskTexture, maskSampler, maskUv - vec2f(0.0, erode.y)).r);
  return confidence;
}
fn personAlpha(maskUv: vec2f, feather: f32, erode: vec2f, threshold: f32) -> f32 {
  let insideMask = all(maskUv >= vec2f(0.0)) && all(maskUv <= vec2f(1.0));
  // The eroded mask keeps the coarse steps of the model output; averaging it over a small
  // diamond turns those steps into a smooth ramp before the feather is applied.
  let smoothing = erode * 0.75;
  var confidence = erodedMask(maskUv, erode) * 2.0;
  confidence += erodedMask(maskUv + vec2f(smoothing.x, smoothing.y), erode);
  confidence += erodedMask(maskUv + vec2f(-smoothing.x, smoothing.y), erode);
  confidence += erodedMask(maskUv + vec2f(smoothing.x, -smoothing.y), erode);
  confidence += erodedMask(maskUv + vec2f(-smoothing.x, -smoothing.y), erode);
  let alpha = smoothstep(threshold - feather, threshold + feather, confidence / 6.0);
  return select(0.0, alpha, insideMask);
}
`;

// 48 bytes, matching the JS-side buffer: a trailing vec3f would pad the struct to 64.
// settings = (edge feather, erode u, erode v, threshold).
const COMPOSITE_PARAMS_STRUCT = /* wgsl */ `
struct CompositeParams {
  maskRow0: vec4f,
  maskRow1: vec4f,
  settings: vec4f,
};
`;

// First rung of the blur ladder: the same five-tap downsample, but every tap is weighted by how
// much background it holds and the weight travels in alpha. Blurring only the background keeps
// the person's colours from bleeding into it, which is what shows as a halo around the outline.
const MASKED_DOWNSAMPLE_FRAGMENT = /* wgsl */ `
${COMPOSITE_PARAMS_STRUCT}
@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;
@group(1) @binding(0) var maskTexture: texture_2d<f32>;
@group(1) @binding(1) var maskSampler: sampler;
@group(2) @binding(0) var<uniform> params: CompositeParams;
${PERSON_ALPHA_FUNCTION}
fn backgroundTap(uv: vec2f) -> vec4f {
  let maskUv = vec2f(dot(params.maskRow0.xyz, vec3f(uv, 1.0)), dot(params.maskRow1.xyz, vec3f(uv, 1.0)));
  let weight = 1.0 - personAlpha(maskUv, params.settings.x, params.settings.yz, params.settings.w);
  return vec4f(textureSample(sourceTexture, sourceSampler, uv).rgb * weight, weight);
}
@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let offset = 1.0 / vec2f(textureDimensions(sourceTexture));
  var color = backgroundTap(input.uv) * 4.0;
  color += backgroundTap(input.uv - offset);
  color += backgroundTap(input.uv + offset);
  color += backgroundTap(input.uv + vec2f(offset.x, -offset.y));
  color += backgroundTap(input.uv - vec2f(offset.x, -offset.y));
  return color / 8.0;
}
`;

const MASK_COMPOSITE_FRAGMENT = /* wgsl */ `
${COMPOSITE_PARAMS_STRUCT}
@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;
@group(1) @binding(0) var backgroundTexture: texture_2d<f32>;
@group(1) @binding(1) var backgroundSampler: sampler;
@group(2) @binding(0) var maskTexture: texture_2d<f32>;
@group(2) @binding(1) var maskSampler: sampler;
@group(3) @binding(0) var<uniform> params: CompositeParams;
${PERSON_ALPHA_FUNCTION}
@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let maskUv = vec2f(
    dot(params.maskRow0.xyz, vec3f(input.uv, 1.0)),
    dot(params.maskRow1.xyz, vec3f(input.uv, 1.0)),
  );
  let alpha = personAlpha(maskUv, params.settings.x, params.settings.yz, params.settings.w);
  // The blurred background is premultiplied by its background weight; dividing it out recovers
  // the true background colour right up to the person's edge.
  let blurred = textureSample(backgroundTexture, backgroundSampler, input.uv);
  let background = blurred.rgb / max(blurred.a, 0.002);
  let source = textureSample(sourceTexture, sourceSampler, input.uv).rgb;
  return vec4f(mix(background, source, alpha), 1.0);
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
  // Outside the model's square, treat as background so the mask edge is not smeared across the strip.
  let insideMask = all(maskUv >= vec2f(0.0)) && all(maskUv <= vec2f(1.0));
  let confidence = select(0.0, textureSample(maskTexture, maskSampler, maskUv).r, insideMask);
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
  readonly downsample: GPURenderPipeline;
  readonly maskedDownsample: GPURenderPipeline;
  readonly upsample: GPURenderPipeline;
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
    downsample: createPipeline(
      device,
      DOWNSAMPLE_FRAGMENT,
      "rgba8unorm",
      [textures],
      "fishjam-video-effect-blur-downsample",
    ),
    maskedDownsample: createPipeline(
      device,
      MASKED_DOWNSAMPLE_FRAGMENT,
      "rgba8unorm",
      [textures, textures, uniforms],
      "fishjam-video-effect-blur-masked-downsample",
    ),
    upsample: createPipeline(
      device,
      UPSAMPLE_FRAGMENT,
      "rgba8unorm",
      [textures],
      "fishjam-video-effect-blur-upsample",
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
  "worklet";
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
  "worklet";
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

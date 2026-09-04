// @ts-nocheck
/** Builds the fixed segmentation graph and records worklet-safe GPU commands. */
import tgpu, {
  common,
  d,
  std,
  type TgpuBindGroup,
  type TgpuBindGroupLayout,
  type TgpuRenderPipeline,
  type TgpuRoot,
  type TgpuUniform,
} from "typegpu";

import { FrameCropParams, initialFrameCropParams } from "./frame";
import type { SegmenterPlan } from "./inference/bundle";
import { createSegmenterDispatches } from "./inference/kernels/dispatches";
import type {
  MaskBuffer as CnnMaskBuffer,
  PackedWeightsBuffer,
  Vec4Buffer,
} from "./inference/kernels/types";
import { WORKGROUP_SIZE } from "./inference/kernels/types";
import {
  recordComputeDispatch,
  type RecordedComputeDispatch,
} from "./nativeRecording";
import {
  initialUpsampleParams,
  MODEL_PIXELS,
  personCorePriorKernel,
  POST_PROCESS_WORKGROUPS,
  PostProcessParams,
  priorLayout,
  temporalAccumulatorKernel,
  temporalLayout,
  UPSAMPLE_WORKGROUP_SIZE,
  upsampleFrameLayout,
  upsampleMaskLayout,
  UpsampleParams,
  upsampleParamsLayout,
  upsampleSamplerLayout,
  upsampleToTextureKernel,
} from "./post-processing/kernels";
import type { MaskBuffer as PostMaskBuffer } from "./post-processing/types";

// ---------------------------------------------------------------------------
// Video preprocess (model-space sampling of the camera external texture).
// Ported from `inference/video-preprocess.ts` but with the layouts inlined here
// so we can record the dispatch + expose the frame layout.
// ---------------------------------------------------------------------------
const MODEL_WIDTH = 256;
const MODEL_HEIGHT = 256;
const MODEL_PIXELS_PREPROCESS = MODEL_WIDTH * MODEL_HEIGHT;
const MODEL_COORD_MASK = MODEL_WIDTH - 1;
const MODEL_COORD_SHIFT = 8;
const MODEL_SIZE = d.vec2f(MODEL_WIDTH, MODEL_HEIGHT);

const videoFrameParamsLayout = tgpu.bindGroupLayout({
  params: { uniform: FrameCropParams },
});
const videoFrameFrameLayout = tgpu.bindGroupLayout({
  frame: { externalTexture: d.textureExternal() },
});
const videoFrameOutputLayout = tgpu.bindGroupLayout({
  sampler: { sampler: "filtering" },
  dst: { storage: d.arrayOf(d.vec4f), access: "mutable" },
});

const videoPreprocessKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})(({ gid }) => {
  "use gpu";
  const i = gid.x;
  if (i >= MODEL_PIXELS_PREPROCESS) {
    return;
  }

  const coord = d.vec2u(
    i & MODEL_COORD_MASK,
    std.bitShiftRight(i, MODEL_COORD_SHIFT),
  );
  const pixel = d.vec2f(coord) + 0.5;
  const cropUv = d.vec2f(MODEL_SIZE.x - pixel.x, pixel.y) / MODEL_SIZE;
  const sourceUv =
    (videoFrameParamsLayout.$.params.cropOrigin +
      cropUv * videoFrameParamsLayout.$.params.cropSize) /
    d.vec2f(videoFrameParamsLayout.$.params.sourceSize);
  const uv =
    videoFrameParamsLayout.$.params.uvTransform * (sourceUv - 0.5) + 0.5;

  const color = std.textureSampleBaseClampToEdge(
    videoFrameFrameLayout.$.frame,
    videoFrameOutputLayout.$.sampler,
    uv,
  );

  videoFrameOutputLayout.$.dst[i] = d.vec4f(color.rgb, 0);
});

// ---------------------------------------------------------------------------
// Composite render (camera + mask + synthetic gradient background) → IOSurface.
// Ported from the example's `compositeFragment`, but the color attachment is the
// IOSurface texture view (supplied per-frame in the worklet), NOT a canvas.
// ---------------------------------------------------------------------------
const PERSON_ALPHA_LOW = 0.35;
const PERSON_ALPHA_HIGH = 0.65;

const compositeFrameLayout = tgpu.bindGroupLayout({
  frame: { externalTexture: d.textureExternal() },
});
const compositeMaskLayout = tgpu.bindGroupLayout({
  mask: { texture: d.texture2d() },
});

// ---------------------------------------------------------------------------
// The fully-flattened, worklet-serializable result of setup.
// ---------------------------------------------------------------------------
export interface SegmentationBundle {
  /** Square edge length of the IOSurface output (composite target). */
  readonly outputSize: number;
  /** The upsampled person-confidence mask consumed by the effects compositor. */
  readonly maskView: GPUTextureView;

  // ---- Per-frame compute dispatches, in execution order. ----
  /** Model-space camera sampling. Has an external-texture frame group. */
  readonly preprocess: RecordedComputeDispatch;
  /** The CNN op chain (conv/dwconv/pool/resize/add/mul/head). All static. */
  readonly cnn: RecordedComputeDispatch[];
  /** Temporal accumulator. Static. */
  readonly temporal: RecordedComputeDispatch;
  /** Person-core prior. Static. */
  readonly prior: RecordedComputeDispatch;
  /** Bilateral edge-aware upsample. Has an external-texture frame group. */
  readonly upsample: RecordedComputeDispatch;

  // ---- Raw natives the worklet needs for the per-frame work. ----
  /** Bind-group layout for the preprocess external-texture group. */
  readonly preprocessFrameLayout: GPUBindGroupLayout;
  /** Bind-group layout for the upsample external-texture group. */
  readonly upsampleFrameLayout: GPUBindGroupLayout;
  /** Uniform buffer for FrameCropParams consumed by the preprocess kernel. */
  readonly preprocessParamsBuffer: GPUBuffer;
  /** Uniform buffer for UpsampleParams consumed by the upsample kernel. */
  readonly upsampleParamsBuffer: GPUBuffer;
  /** Uniform buffer for PostProcessParams (temporal `initialized` flag). */
  readonly postProcessParamsBuffer: GPUBuffer;

  // ---- Composite render (replayed natively in the worklet). ----
  readonly compositePipeline: GPURenderPipeline;
  readonly compositeFrameLayout: GPUBindGroupLayout;
  /** Static composite group(s): the mask texture + the composite uniform. */
  readonly compositeStaticGroups: { index: number; bindGroup: GPUBindGroup }[];
  /** Group index where the worklet binds the composite external-texture group. */
  readonly compositeFrameGroupIndex: number;
  /** Uniform buffer for the composite FrameCropParams (crop/orientation). */
  readonly compositeParamsBuffer: GPUBuffer;
}

/**
 * Builds the full segmentation pipeline on the JS thread and flattens it to raw
 * natives. Call once, after the GPUDevice is ready.
 */
export function buildSegmentationBundle(
  root: TgpuRoot,
  plan: SegmenterPlan,
  outputSize: number,
): SegmentationBundle {
  // ---- CNN buffers + weights (from `segmenter.ts`). ----
  const cnnMask = root
    .createBuffer(d.arrayOf(d.f32, plan.slotSizesVec4[1]))
    .$usage("storage") as CnnMaskBuffer;
  const slots = plan.slotSizesVec4.map((n, slot) =>
    slot === 1
      ? (cnnMask as unknown as Vec4Buffer)
      : (root
          .createBuffer(d.arrayOf(d.vec4f, n))
          .$usage("storage") as Vec4Buffer),
  );
  const inputSlot = slots[0];
  const weights = root
    .createBuffer(
      d.arrayOf(d.u32, plan.weights.byteLength / Uint32Array.BYTES_PER_ELEMENT),
    )
    .$usage("storage") as PackedWeightsBuffer;
  weights.write(plan.weights);

  const cnnHandles = createSegmenterDispatches(
    root,
    plan.dispatches,
    slots,
    cnnMask,
    weights,
  );

  // ---- Video preprocess (`video-preprocess.ts`). ----
  const preprocessWorkgroups = Math.ceil(
    MODEL_PIXELS_PREPROCESS / WORKGROUP_SIZE,
  );
  const preprocessPipeline = root.createComputePipeline({
    compute: videoPreprocessKernel,
  });
  const preprocessParamsTgpu = root
    .createBuffer(FrameCropParams, initialFrameCropParams)
    .$usage("uniform");
  const preprocessSampler = root.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });
  const preprocessParamsGroup = root.createBindGroup(videoFrameParamsLayout, {
    params: preprocessParamsTgpu,
  });
  const preprocessOutputGroup = root.createBindGroup(videoFrameOutputLayout, {
    sampler: preprocessSampler,
    dst: inputSlot,
  });

  // ---- Post-process buffers + pipelines (`processor.ts`). ----
  const historyLogits = root
    .createBuffer(d.arrayOf(d.f32, MODEL_PIXELS))
    .$usage("storage") as PostMaskBuffer;
  const temporalMask = root
    .createBuffer(d.arrayOf(d.f32, MODEL_PIXELS))
    .$usage("storage") as PostMaskBuffer;
  const priorMask = root
    .createBuffer(d.arrayOf(d.f32, MODEL_PIXELS))
    .$usage("storage") as PostMaskBuffer;

  const postProcessParamsTgpu = root
    .createBuffer(PostProcessParams, { initialized: 0 })
    .$usage("uniform");
  const upsampleParamsTgpu = root
    .createBuffer(UpsampleParams, initialUpsampleParams)
    .$usage("uniform");
  const upsampleSampler = root.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  const temporalPipeline = root.createComputePipeline({
    compute: temporalAccumulatorKernel,
  });
  const temporalGroup = root.createBindGroup(temporalLayout, {
    params: postProcessParamsTgpu,
    raw: cnnMask as unknown as PostMaskBuffer,
    historyLogits,
    filtered: temporalMask,
  });
  const priorPipeline = root.createComputePipeline({
    compute: personCorePriorKernel,
  });
  const priorGroup = root.createBindGroup(priorLayout, {
    src: temporalMask,
    dst: priorMask,
  });

  // Mask output texture (rgba16float storage+sampled), sized to the output.
  const maskTexture = root
    .createTexture({
      size: [outputSize, outputSize],
      format: "rgba16float" as const,
    })
    .$usage("storage", "sampled");
  const maskSampleView = maskTexture.createView();
  const maskStorageView = maskTexture.createView(
    d.textureStorage2d("rgba16float", "write-only"),
  );

  const upsamplePipeline = root.createComputePipeline({
    compute: upsampleToTextureKernel,
  });
  const upsampleParamsGroup = root.createBindGroup(upsampleParamsLayout, {
    params: upsampleParamsTgpu,
  });
  const upsampleSamplerGroup = root.createBindGroup(upsampleSamplerLayout, {
    sampler: upsampleSampler,
  });
  const upsampleMaskGroup = root.createBindGroup(upsampleMaskLayout, {
    src: priorMask,
    output: maskStorageView,
  });
  const upsampleWorkgroupsX = Math.ceil(outputSize / UPSAMPLE_WORKGROUP_SIZE);
  const upsampleWorkgroupsY = Math.ceil(outputSize / UPSAMPLE_WORKGROUP_SIZE);

  // ---- Composite render pipeline. ----
  const compositeUniform = root.createUniform(
    FrameCropParams,
    initialFrameCropParams,
  );
  const compositeSampler = root.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  const sampleMask = (uv: d.v2f) => {
    "use gpu";
    return std.textureSample(compositeMaskLayout.$.mask, compositeSampler.$, uv)
      .r;
  };
  const sampleFeatheredMask = (uv: d.v2f) => {
    "use gpu";
    const texel =
      1 / d.vec2f(std.textureDimensions(compositeMaskLayout.$.mask));
    const center = sampleMask(uv) * 4;
    const cardinal =
      (sampleMask(uv + d.vec2f(texel.x, 0)) +
        sampleMask(uv - d.vec2f(texel.x, 0)) +
        sampleMask(uv + d.vec2f(0, texel.y)) +
        sampleMask(uv - d.vec2f(0, texel.y))) *
      2;
    const diagonal =
      sampleMask(uv + texel) +
      sampleMask(uv - texel) +
      sampleMask(uv + d.vec2f(texel.x, -texel.y)) +
      sampleMask(uv + d.vec2f(-texel.x, texel.y));
    return (center + cardinal + diagonal) * 0.0625;
  };

  const compositeFragment = tgpu.fragmentFn({
    in: { uv: d.vec2f },
    out: d.vec4f,
  })(({ uv }) => {
    "use gpu";
    const cropUv = d.vec2f(1 - uv.x, uv.y);
    const sourcePixel =
      compositeUniform.$.cropOrigin + cropUv * compositeUniform.$.cropSize;
    const sourceUv = sourcePixel / d.vec2f(compositeUniform.$.sourceSize);
    const cameraUv = compositeUniform.$.uvTransform * (sourceUv - 0.5) + 0.5;
    const cameraColor = std.textureSampleBaseClampToEdge(
      compositeFrameLayout.$.frame,
      compositeSampler.$,
      cameraUv,
    );
    const personMask = sampleFeatheredMask(uv);
    const personAlpha = std.smoothstep(
      PERSON_ALPHA_LOW,
      PERSON_ALPHA_HIGH,
      personMask,
    );
    const vertical = std.mix(
      d.vec3f(0.09, 0.2, 0.62),
      d.vec3f(0.96, 0.3, 0.45),
      uv.y,
    );
    const gradient = std.mix(vertical, d.vec3f(1, 0.84, 0.38), uv.x * 0.35);
    return d.vec4f(std.mix(gradient, cameraColor.rgb, personAlpha), 1);
  });

  const compositePipelineTgpu = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: compositeFragment,
    // IOSurface custom-track textures are imported as bgra8unorm (see
    // WebGPUVideoTrack.create); the composite color target MUST match.
    targets: { format: "bgra8unorm" as GPUTextureFormat },
  });
  const compositeMaskGroup = root.createBindGroup(compositeMaskLayout, {
    mask: maskSampleView,
  });

  // ---- RECORD every compute dispatch into raw natives. ----
  const preprocess = recordComputeDispatch(
    root,
    { pipeline: preprocessPipeline },
    [preprocessParamsGroup, preprocessOutputGroup],
    { x: preprocessWorkgroups },
    // Frame group index is derived as the single index not covered by the
    // recorded static groups (recordedCount + 1 total); no hardcoded count.
    { frameLayout: videoFrameFrameLayout },
  );

  const cnn = cnnHandles.map((handle) =>
    recordComputeDispatch(
      root,
      { pipeline: handle.pipeline },
      [handle.bindGroup],
      {
        x: handle.workgroups,
      },
    ),
  );

  const temporal = recordComputeDispatch(
    root,
    { pipeline: temporalPipeline },
    [temporalGroup],
    { x: POST_PROCESS_WORKGROUPS },
  );
  const prior = recordComputeDispatch(
    root,
    { pipeline: priorPipeline },
    [priorGroup],
    { x: POST_PROCESS_WORKGROUPS },
  );
  const upsample = recordComputeDispatch(
    root,
    { pipeline: upsamplePipeline },
    [upsampleParamsGroup, upsampleSamplerGroup, upsampleMaskGroup],
    { x: upsampleWorkgroupsX, y: upsampleWorkgroupsY },
    { frameLayout: upsampleFrameLayout },
  );

  // ---- Flatten composite render bind groups by recording the render apply. ----
  const composite = recordRenderDispatch(
    root,
    compositePipelineTgpu,
    compositeMaskGroup,
    compositeUniform,
    /* frameLayout */ compositeFrameLayout,
  );

  return {
    outputSize,
    maskView: root.unwrap(maskSampleView),
    preprocess,
    cnn,
    temporal,
    prior,
    upsample,
    preprocessFrameLayout: root.unwrap(videoFrameFrameLayout),
    upsampleFrameLayout: root.unwrap(upsampleFrameLayout),
    preprocessParamsBuffer: root.unwrap(preprocessParamsTgpu),
    upsampleParamsBuffer: root.unwrap(upsampleParamsTgpu),
    postProcessParamsBuffer: root.unwrap(postProcessParamsTgpu),
    compositePipeline: composite.pipeline,
    compositeFrameLayout: root.unwrap(compositeFrameLayout),
    compositeStaticGroups: composite.staticGroups,
    compositeFrameGroupIndex: composite.frameGroupIndex,
    compositeParamsBuffer: root.unwrap(compositeUniform.buffer),
  };
}

interface RecordedRenderDispatch {
  pipeline: GPURenderPipeline;
  staticGroups: { index: number; bindGroup: GPUBindGroup }[];
  frameGroupIndex: number;
}

function recordRenderDispatch(
  root: TgpuRoot,
  pipeline: TgpuRenderPipeline,
  maskGroup: TgpuBindGroup,
  // The composite uniform is bound via its own catch-all/uniform group; tgpu
  // resolves it automatically, so we only supply the mask group explicitly. The
  // uniform's group is recorded as a static group too.
  _compositeUniform: TgpuUniform<typeof FrameCropParams>,
  _frameLayout: TgpuBindGroupLayout,
): RecordedRenderDispatch {
  const captured: {
    pipeline: GPURenderPipeline | null;
    groups: { index: number; bindGroup: GPUBindGroup }[];
  } = { pipeline: null, groups: [] };

  // A proxy carrying `executeBundles` + `draw` is recognized by tgpu as a
  // GPURenderPassEncoder, so `pipeline.with(proxy)` takes the external-render-
  // encoder path: tgpu calls `_applyRenderState(proxy)` (recording setPipeline +
  // each setBindGroup, including the auto-bound composite uniform "catch-all"
  // group) and then `proxy.draw(...)`. No real attachment / beginRenderPass is
  // needed. The per-frame external-texture group is intentionally NOT supplied,
  // so it surfaces as the one missing group index.
  const proxyRenderPass = {
    executeBundles() {},
    setPipeline(p: GPURenderPipeline) {
      captured.pipeline = p;
    },
    setBindGroup(index: number, bindGroup: GPUBindGroup) {
      captured.groups.push({ index, bindGroup });
    },
    setVertexBuffer() {},
    setIndexBuffer() {},
    setStencilReference() {},
    draw() {},
    pushDebugGroup() {},
    popDebugGroup() {},
    insertDebugMarker() {},
  } as unknown as GPURenderPassEncoder;

  // The external-texture group is intentionally absent, so tgpu records the
  // pipeline + static groups (mask + the auto-bound composite-uniform catch-all)
  // and then throws `MissingBindGroupsError`. That throw is expected here.
  try {
    pipeline.with(proxyRenderPass).with(maskGroup).draw(3);
  } catch {
    // expected: missing external-texture group (bound per-frame in the worklet)
  }

  if (!captured.pipeline) {
    throw new Error("recordRenderDispatch: tgpu did not call setPipeline");
  }
  // Exactly one external-texture group is missing, so the total group count is
  // recordedCount + 1; the frame group is the single uncovered index.
  const used = new Set(captured.groups.map((g) => g.index));
  const total = captured.groups.length + 1;
  let frameGroupIndex = -1;
  for (let index = 0; index < total; index++) {
    if (!used.has(index)) {
      frameGroupIndex = index;
      break;
    }
  }
  if (frameGroupIndex < 0) {
    throw new Error(
      "recordRenderDispatch: could not locate external-texture group",
    );
  }
  return {
    pipeline: captured.pipeline,
    staticGroups: captured.groups,
    frameGroupIndex,
  };
}

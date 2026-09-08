// @ts-nocheck
/** Builds the fixed segmentation graph and records worklet-safe GPU commands. */
import tgpu, { d, std, type TgpuRoot } from "typegpu";

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
// Video preprocess: samples the frame the effect composites (an upright RGBA
// `texture_2d`) into model space. The frame layout is exposed so the per-frame
// bind group can be built on the camera thread.
// ---------------------------------------------------------------------------
const MODEL_WIDTH = 256;
const MODEL_HEIGHT = 256;
const MODEL_PIXELS_PREPROCESS = MODEL_WIDTH * MODEL_HEIGHT;
const MODEL_COORD_MASK = MODEL_WIDTH - 1;
const MODEL_COORD_SHIFT = 8;
const MODEL_SIZE = d.vec2f(MODEL_WIDTH, MODEL_HEIGHT);

const videoFrameParamsLayout = tgpu
  .bindGroupLayout({
    params: { uniform: FrameCropParams },
  })
  .$idx(0);
const videoFrameFrameLayout = tgpu
  .bindGroupLayout({
    frame: { texture: d.texture2d(d.f32) },
  })
  .$idx(1);
const videoFrameOutputLayout = tgpu
  .bindGroupLayout({
    sampler: { sampler: "filtering" },
    dst: { storage: d.arrayOf(d.vec4f), access: "mutable" },
  })
  .$idx(2);

const videoPreprocessKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})(({ gid }) => {
  "use gpu";
  const i = gid.x;
  if (i >= MODEL_PIXELS_PREPROCESS) {
    return;
  }

  const coord = d.vec2u(i & MODEL_COORD_MASK, i >>> MODEL_COORD_SHIFT);
  const cropUv = (d.vec2f(coord) + 0.5) / MODEL_SIZE;
  const sourceUv =
    (videoFrameParamsLayout.$.params.cropOrigin +
      cropUv * videoFrameParamsLayout.$.params.cropSize) /
    d.vec2f(videoFrameParamsLayout.$.params.sourceSize);
  const uv =
    videoFrameParamsLayout.$.params.uvTransform * (sourceUv - 0.5) + 0.5;

  const color = std.textureSampleLevel(
    videoFrameFrameLayout.$.frame,
    videoFrameOutputLayout.$.sampler,
    uv,
    0,
  );

  videoFrameOutputLayout.$.dst[i] = d.vec4f(color.rgb, 0);
});

// ---------------------------------------------------------------------------
// The fully-flattened, worklet-serializable result of setup.
// ---------------------------------------------------------------------------
export interface SegmentationBundle {
  /** Size of the mask texture: the composite target's size, so the mask is per output pixel. */
  readonly outputWidth: number;
  readonly outputHeight: number;
  /** The upsampled person-confidence mask consumed by the effects compositor. */
  readonly maskView: GPUTextureView;

  // ---- Per-frame compute dispatches, in execution order. ----
  /** Model-space frame sampling. Has a frame-texture group. */
  readonly preprocess: RecordedComputeDispatch;
  /** The CNN op chain (conv/dwconv/pool/resize/add/mul/head). All static. */
  readonly cnn: RecordedComputeDispatch[];
  /** Temporal accumulator. Static. */
  readonly temporal: RecordedComputeDispatch;
  /** Person-core prior. Static. */
  readonly prior: RecordedComputeDispatch;
  /** Bilateral edge-aware upsample. Has a frame-texture group. */
  readonly upsample: RecordedComputeDispatch;

  // ---- Raw natives the worklet needs for the per-frame work. ----
  /** Bind-group layout for the preprocess frame-texture group. */
  readonly preprocessFrameLayout: GPUBindGroupLayout;
  /** Bind-group layout for the upsample frame-texture group. */
  readonly upsampleFrameLayout: GPUBindGroupLayout;
  /** Uniform buffer for FrameCropParams consumed by the preprocess kernel. */
  readonly preprocessParamsBuffer: GPUBuffer;
  /** Uniform buffer for UpsampleParams consumed by the upsample kernel. */
  readonly upsampleParamsBuffer: GPUBuffer;
  /** Uniform buffer for PostProcessParams (temporal `initialized` flag). */
  readonly postProcessParamsBuffer: GPUBuffer;
}

/**
 * Builds the full segmentation pipeline on the JS thread and flattens it to raw
 * natives. Call once, after the GPUDevice is ready.
 */
export function buildSegmentationBundle(
  root: TgpuRoot,
  plan: SegmenterPlan,
  outputWidth: number,
  outputHeight: number,
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

  // Mask output texture (rgba16float storage+sampled), one texel per output pixel.
  const maskTexture = root
    .createTexture({
      size: [outputWidth, outputHeight],
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
  const upsampleWorkgroupsX = Math.ceil(outputWidth / UPSAMPLE_WORKGROUP_SIZE);
  const upsampleWorkgroupsY = Math.ceil(outputHeight / UPSAMPLE_WORKGROUP_SIZE);

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

  return {
    outputWidth,
    outputHeight,
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
  };
}

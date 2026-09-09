/// <reference types="@webgpu/types" preserve="true" />

import tgpu from "typegpu";

import type {
  PersonMask,
  PersonSegmentationFrameKernel,
  PersonSegmentationKernelState,
  PersonSegmentationProvider,
  PersonSegmentationSession,
  SegmentationContext,
  SegmentationInput,
} from "../../core/types";
import {
  computeFullFrameCrop,
  type FrameCrop,
  packFrameCropParams,
  packUpsampleParams,
} from "./internal/frameParams";
import { parseSegmenterPlan } from "./internal/inference/bundle";
import {
  buildSegmentationBundle,
  type SegmentationBundle,
} from "./internal/segmentationPipeline";

const DEFAULT_MODEL_URL = new URL(
  "../../../../assets/selfie_segmenter.ssgbin",
  import.meta.url,
).toString();

const PROVIDER_ID = "typegpu-selfie-segmentation-experimental";

export interface TypeGpuPersonSegmentationOptions {
  /** CDN or application asset URL. The default points at this package's bundled model. */
  readonly modelUrl?: string;
}

// Plain data only (numbers plus GPU objects): the state is copied onto the camera thread's
// worklet runtime, where `offerFrame` and `latestMask` run. TypeGPU's own objects stay on the
// JS thread; `buildSegmentationBundle` has already flattened them into raw WebGPU handles.
interface TypeGpuState extends PersonSegmentationKernelState {
  readonly device: GPUDevice;
  readonly bundle: SegmentationBundle;
  timestampUs: number;
  initialized: boolean;
  /** The crop the latest mask was computed for; null until the first frame. */
  crop: FrameCrop | null;
}

/**
 * Experimental fixed-graph MediaPipe Selfie Segmentation provider. It encodes
 * inference into the caller's command encoder and returns a GPU-only mask.
 */
export function typeGpuPersonSegmentation(
  options: TypeGpuPersonSegmentationOptions = {},
): PersonSegmentationProvider {
  return {
    id: PROVIDER_ID,
    input: "gpu-texture",
    prepare: async (context) => createTypeGpuSession(context, options),
  };
}

async function createTypeGpuSession(
  context: SegmentationContext,
  options: TypeGpuPersonSegmentationOptions,
): Promise<PersonSegmentationSession> {
  const buffer = await loadModel(options.modelUrl ?? DEFAULT_MODEL_URL);
  const plan = parseSegmenterPlan(buffer);
  const root = await tgpu.initFromDevice({ device: context.device });
  const bundle = buildSegmentationBundle(
    root,
    plan,
    context.outputWidth,
    context.outputHeight,
  );
  const state: TypeGpuState = {
    providerId: PROVIDER_ID,
    device: context.device,
    bundle,
    timestampUs: Number.NEGATIVE_INFINITY,
    initialized: false,
    crop: null,
  };
  const frameKernel: PersonSegmentationFrameKernel = {
    state,
    offer: offerFrame,
    latest: latestMask,
    reset: resetTimeline,
  };
  // Disposal is tracked here, not in `state`, which may already be copied to a worklet runtime.
  let disposed = false;
  return {
    frameKernel,
    offer: (input) => {
      if (!disposed) offerFrame(state, input);
    },
    latest: (renderTimestampUs) =>
      disposed ? null : latestMask(state, renderTimestampUs),
    reset: () => {
      if (!disposed) resetTimeline(state);
    },
    dispose: () => {
      if (disposed) return;
      disposed = true;
      root.destroy();
    },
  };
}

// Worklet helpers are captured by value when a caller is defined, so they must precede
// their callers in this module.
function dispatch(
  pass: GPUComputePassEncoder,
  operation: SegmentationBundle["preprocess"],
  frameGroup?: GPUBindGroup,
): void {
  "worklet";
  pass.setPipeline(operation.pipeline);
  for (const group of operation.staticGroups)
    pass.setBindGroup(group.index, group.bindGroup);
  if (frameGroup != null)
    pass.setBindGroup(operation.frameGroupIndex, frameGroup);
  pass.dispatchWorkgroups(operation.workgroupsX, operation.workgroupsY);
}

function encodeMask(
  bundle: SegmentationBundle,
  device: GPUDevice,
  frame: GPUTextureView,
  encoder: GPUCommandEncoder,
): void {
  "worklet";
  const preprocessFrameGroup = device.createBindGroup({
    layout: bundle.preprocessFrameLayout,
    entries: [{ binding: 0, resource: frame }],
  });
  const upsampleFrameGroup = device.createBindGroup({
    layout: bundle.upsampleFrameLayout,
    entries: [{ binding: 0, resource: frame }],
  });
  const pass = encoder.beginComputePass();
  dispatch(pass, bundle.preprocess, preprocessFrameGroup);
  for (const cnn of bundle.cnn) dispatch(pass, cnn);
  dispatch(pass, bundle.temporal);
  dispatch(pass, bundle.prior);
  dispatch(pass, bundle.upsample, upsampleFrameGroup);
  pass.end();
}

function offerFrame(
  kernelState: PersonSegmentationKernelState,
  input: SegmentationInput,
): void {
  "worklet";
  const state = kernelState as TypeGpuState;
  if (input.kind !== "gpu-texture") return;
  if (input.timestampUs < state.timestampUs) return;
  const crop = computeFullFrameCrop(input.width, input.height);
  state.crop = crop;
  const device = state.device;
  device.queue.writeBuffer(
    state.bundle.preprocessParamsBuffer,
    0,
    packFrameCropParams(crop),
  );
  device.queue.writeBuffer(
    state.bundle.upsampleParamsBuffer,
    0,
    packUpsampleParams(crop, true),
  );
  device.queue.writeBuffer(
    state.bundle.postProcessParamsBuffer,
    0,
    new Uint32Array([state.initialized ? 1 : 0]),
  );
  encodeMask(state.bundle, device, input.texture, input.commandEncoder);
  state.initialized = true;
  state.timestampUs = input.timestampUs;
}

/**
 * Maps a source UV to the mask texture's UV: the inverse of the crop the model looked at
 * (`videoPreprocessKernel` samples `cropOrigin + uv * cropSize` of the source).
 */
function maskTransformFor(crop: FrameCrop): Float32Array {
  "worklet";
  const scaleX = crop.sourceWidth / crop.cropSizeX;
  const scaleY = crop.sourceHeight / crop.cropSizeY;
  const offsetX = -crop.cropOriginX / crop.cropSizeX;
  const offsetY = -crop.cropOriginY / crop.cropSizeY;
  return new Float32Array([scaleX, 0, offsetX, 0, scaleY, offsetY]);
}

function latestMask(
  kernelState: PersonSegmentationKernelState,
  renderTimestampUs: number,
): PersonMask | null {
  "worklet";
  const state = kernelState as TypeGpuState;
  if (state.crop == null || state.timestampUs > renderTimestampUs) return null;
  return {
    texture: state.bundle.maskView,
    timestampUs: state.timestampUs,
    sourceUvToMaskUv: maskTransformFor(state.crop),
  };
}

function resetTimeline(kernelState: PersonSegmentationKernelState): void {
  "worklet";
  const state = kernelState as TypeGpuState;
  state.initialized = false;
  state.timestampUs = Number.NEGATIVE_INFINITY;
}

async function loadModel(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(
      `TypeGPU segmentation model fetch failed: ${response.status}.`,
    );
  return response.arrayBuffer();
}

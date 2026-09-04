/// <reference types="@webgpu/types" preserve="true" />

import tgpu from "typegpu";

import type {
  PersonMask,
  PersonSegmentationProvider,
  PersonSegmentationSession,
  SegmentationContext,
  SegmentationInput,
} from "../../core/types";
import {
  computeSquareCrop,
  packFrameCropParams,
  packUpsampleParams,
} from "./internal/frameParams";
import { parseSegmenterPlan } from "./internal/inference/bundle";
import {
  buildSegmentationBundle,
  type SegmentationBundle,
} from "./internal/segmentationPipeline";

const DEFAULT_MODEL_URL = new URL(
  "../../../assets/selfie_segmenter.ssgbin",
  import.meta.url,
).toString();

export interface TypeGpuPersonSegmentationOptions {
  /** CDN or application asset URL. The default points at this package's bundled model. */
  readonly modelUrl?: string;
}

/**
 * Experimental fixed-graph MediaPipe Selfie Segmentation provider. It encodes
 * inference into the caller's command encoder and returns a GPU-only mask.
 */
export function typeGpuPersonSegmentation(
  options: TypeGpuPersonSegmentationOptions = {},
): PersonSegmentationProvider {
  return {
    id: "typegpu-selfie-segmentation-experimental",
    input: "gpu-texture",
    prepare: async (context) => {
      const plan = parseSegmenterPlan(
        await loadModel(options.modelUrl ?? DEFAULT_MODEL_URL),
      );
      const root = await tgpu.initFromDevice({ device: context.device });
      const bundle = buildSegmentationBundle(
        root,
        plan,
        Math.max(context.outputWidth, context.outputHeight),
      );
      return new TypeGpuSession(context, bundle, root);
    },
  };
}

class TypeGpuSession implements PersonSegmentationSession {
  private readonly context: SegmentationContext;
  private readonly bundle: SegmentationBundle;
  private readonly root: Awaited<ReturnType<typeof tgpu.initFromDevice>>;
  private timestampUs = Number.NEGATIVE_INFINITY;
  private initialized = false;
  private disposed = false;

  constructor(
    context: SegmentationContext,
    bundle: SegmentationBundle,
    root: Awaited<ReturnType<typeof tgpu.initFromDevice>>,
  ) {
    this.context = context;
    this.bundle = bundle;
    this.root = root;
  }

  offer(input: SegmentationInput): void {
    if (
      this.disposed ||
      input.kind !== "gpu-texture" ||
      input.externalTexture == null
    )
      return;
    if (input.timestampUs < this.timestampUs) return;
    const crop = computeSquareCrop(input.width, input.height, 1, 0, 0, 1);
    const device = this.context.device;
    device.queue.writeBuffer(
      this.bundle.preprocessParamsBuffer,
      0,
      packFrameCropParams(crop),
    );
    device.queue.writeBuffer(
      this.bundle.upsampleParamsBuffer,
      0,
      packUpsampleParams(crop, true),
    );
    device.queue.writeBuffer(
      this.bundle.postProcessParamsBuffer,
      0,
      new Uint32Array([this.initialized ? 1 : 0]),
    );
    encodeMask(
      this.bundle,
      device,
      input.externalTexture,
      input.commandEncoder,
    );
    this.initialized = true;
    this.timestampUs = input.timestampUs;
  }

  latest(renderTimestampUs: number): PersonMask | null {
    return this.timestampUs <= renderTimestampUs
      ? {
          texture: this.bundle.maskView,
          timestampUs: this.timestampUs,
          sourceUvToMaskUv: new Float32Array([1, 0, 0, 0, 1, 0]),
        }
      : null;
  }

  reset(): void {
    this.initialized = false;
    this.timestampUs = Number.NEGATIVE_INFINITY;
  }
  dispose(): void {
    if (!this.disposed) {
      this.disposed = true;
      this.root.destroy();
    }
  }
}

function encodeMask(
  bundle: SegmentationBundle,
  device: GPUDevice,
  frame: GPUExternalTexture,
  encoder: GPUCommandEncoder,
): void {
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

function dispatch(
  pass: GPUComputePassEncoder,
  operation: SegmentationBundle["preprocess"],
  frameGroup?: GPUBindGroup,
): void {
  pass.setPipeline(operation.pipeline);
  for (const group of operation.staticGroups)
    pass.setBindGroup(group.index, group.bindGroup);
  if (frameGroup != null)
    pass.setBindGroup(operation.frameGroupIndex, frameGroup);
  pass.dispatchWorkgroups(operation.workgroupsX, operation.workgroupsY);
}

async function loadModel(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(
      `TypeGPU segmentation model fetch failed: ${response.status}.`,
    );
  return response.arrayBuffer();
}

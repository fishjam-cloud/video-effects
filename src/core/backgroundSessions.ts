/// <reference types="@webgpu/types" preserve="true" />

import { clamp, MASK_MAX_AGE_US } from "./constants";
import { type LoadedImageTexture, loadImageTexture } from "./image";
import {
  type BackgroundPipelines,
  createBackgroundPipelines,
  createSampleTexture,
  createTextureBindGroup,
  createUniformBindGroup,
  createUniformBuffer,
  drawFullscreen,
} from "./pipelines";
import type {
  BackgroundBlurFrameOptions,
  BackgroundBlurOptions,
  BackgroundImageFrameOptions,
  BackgroundImageOptions,
  PersonMask,
  PersonSegmentationFrameKernel,
  PersonSegmentationProvider,
  PersonSegmentationSession,
  SegmentationInput,
  VideoEffectContext,
  VideoEffectFrame,
  VideoEffectFrameKernel,
  VideoEffectKernelState,
  VideoEffectSession,
} from "./types";

const MAX_BLUR_LEVELS = 4;
// Outline treatment (see the personAlpha shader function): the mask is eroded by this many
// texels, then blended with this threshold and feather.
const EDGE_ERODE_PIXELS = 3;
const EDGE_THRESHOLD = 0.55;
const EDGE_FEATHER = 0.12;
// A radius of this many source pixels is one resolution halving; each extra halving doubles it.
const BLUR_LEVEL_BASE_PIXELS = 3;

// Sessions are plain state plus worklet functions, not classes: a class instance cannot be
// copied onto a worklet runtime, and mobile encodes every frame on the camera thread. All
// per-frame work below takes the state explicitly so that a caller holding one copy of it can
// drive `offer` and `encode` against the same data.

interface CommonState extends VideoEffectKernelState {
  readonly device: GPUDevice;
  readonly width: number;
  readonly height: number;
  readonly pipelines: BackgroundPipelines;
  readonly compositeParams: GPUBuffer;
  readonly compositeParamsGroup: GPUBindGroup;
  segmentation: PersonSegmentationFrameKernel | null;
  sourceView: GPUTextureView | null;
  sourceGroup: GPUBindGroup | null;
  maskView: GPUTextureView | null;
  maskGroup: GPUBindGroup | null;
}

/** One rung of the blur's resolution ladder: half the size of the rung above it. */
interface BlurLevel {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  readonly width: number;
  readonly height: number;
  readonly group: GPUBindGroup;
}

interface BlurState extends CommonState {
  readonly effectId: "fishjam.background-blur";
  /** Index 0 is half resolution; every next level halves again. */
  readonly blurLevels: BlurLevel[];
}

interface ImageState extends CommonState {
  readonly effectId: "fishjam.background-image";
  readonly imageParams: GPUBuffer;
  readonly imageParamsGroup: GPUBindGroup;
  image: LoadedImageTexture | null;
  imageGroup: GPUBindGroup | null;
}

// ---------------------------------------------------------------------------
// Per-frame work shared by both effects (worklets).
// ---------------------------------------------------------------------------

function currentMask(
  segmentation: PersonSegmentationFrameKernel | null,
  timestampUs: number,
): PersonMask | null {
  "worklet";
  if (segmentation == null) return null;
  const mask = segmentation.latest(segmentation.state, timestampUs);
  return mask != null && timestampUs - mask.timestampUs <= MASK_MAX_AGE_US
    ? mask
    : null;
}

function writeMaskParams(
  device: GPUDevice,
  buffer: GPUBuffer,
  mask: PersonMask,
  edgeFeather: number,
  erodeU: number,
  erodeV: number,
  threshold: number,
): void {
  "worklet";
  const transform = mask.sourceUvToMaskUv;
  device.queue.writeBuffer(
    buffer,
    0,
    new Float32Array([
      transform[0] ?? 1,
      transform[1] ?? 0,
      transform[2] ?? 0,
      0,
      transform[3] ?? 0,
      transform[4] ?? 1,
      transform[5] ?? 0,
      0,
      edgeFeather,
      erodeU,
      erodeV,
      threshold,
    ]),
  );
}

function sourceGroupFor(
  state: CommonState,
  source: GPUTextureView,
): GPUBindGroup {
  "worklet";
  if (state.sourceGroup == null || state.sourceView !== source) {
    state.sourceView = source;
    state.sourceGroup = createTextureBindGroup(
      state.device,
      state.pipelines.textureLayout,
      source,
      state.pipelines.sampler,
    );
  }
  return state.sourceGroup;
}

function maskGroupFor(state: CommonState, mask: PersonMask): GPUBindGroup {
  "worklet";
  if (state.maskGroup == null || state.maskView !== mask.texture) {
    state.maskView = mask.texture;
    state.maskGroup = createTextureBindGroup(
      state.device,
      state.pipelines.textureLayout,
      mask.texture,
      state.pipelines.sampler,
    );
  }
  return state.maskGroup;
}

function encodeCopy(state: CommonState, frame: VideoEffectFrame): void {
  "worklet";
  drawFullscreen(frame.commandEncoder, frame.output, state.pipelines.copy, [
    sourceGroupFor(state, frame.source),
  ]);
}

function offerToSegmentation(
  state: VideoEffectKernelState,
  input: SegmentationInput,
): void {
  "worklet";
  const common = state as CommonState;
  if (common.segmentation == null) return;
  common.segmentation.offer(common.segmentation.state, input);
}

function resetSegmentation(state: VideoEffectKernelState): void {
  "worklet";
  const common = state as CommonState;
  if (common.segmentation == null) return;
  common.segmentation.reset(common.segmentation.state);
}

// ---------------------------------------------------------------------------
// Background blur.
// ---------------------------------------------------------------------------

/** How many resolution halvings a blur radius (in source pixels) calls for. */
function blurLevelsFor(radius: number, maxLevels: number): number {
  "worklet";
  if (radius <= 0) return 0;
  const levels = Math.round(Math.log2(radius / BLUR_LEVEL_BASE_PIXELS));
  return Math.min(maxLevels, Math.max(1, levels));
}

function encodeBlurFrame(
  state: VideoEffectKernelState,
  frame: VideoEffectFrame,
  options: BackgroundBlurFrameOptions,
): void {
  "worklet";
  const blur = state as BlurState;
  const mask =
    options.enabled === false
      ? null
      : currentMask(blur.segmentation, frame.timestampUs);
  const levels = blurLevelsFor(
    clamp(options.radius, 0, 40, 18),
    blur.blurLevels.length,
  );
  if (mask == null || levels === 0) {
    encodeCopy(blur, frame);
    return;
  }

  const sourceGroup = sourceGroupFor(blur, frame.source);
  const maskGroup = maskGroupFor(blur, mask);
  writeMaskParams(
    blur.device,
    blur.compositeParams,
    mask,
    clamp(options.edgeFeather, 0, 0.5, EDGE_FEATHER),
    EDGE_ERODE_PIXELS / blur.width,
    EDGE_ERODE_PIXELS / blur.height,
    EDGE_THRESHOLD,
  );

  // Dual Kawase: walk down the resolution ladder, then back up. The blur radius doubles with
  // every level; the composite reads the half-resolution top rung with bilinear filtering. The
  // first rung keeps only the background (see the masked downsample shader).
  drawFullscreen(
    frame.commandEncoder,
    blur.blurLevels[0].view,
    blur.pipelines.maskedDownsample,
    [sourceGroup, maskGroup, blur.compositeParamsGroup],
  );
  for (let index = 1; index < levels; index += 1) {
    drawFullscreen(
      frame.commandEncoder,
      blur.blurLevels[index].view,
      blur.pipelines.downsample,
      [blur.blurLevels[index - 1].group],
    );
  }
  for (let index = levels - 1; index >= 1; index -= 1) {
    drawFullscreen(
      frame.commandEncoder,
      blur.blurLevels[index - 1].view,
      blur.pipelines.upsample,
      [blur.blurLevels[index].group],
    );
  }

  drawFullscreen(
    frame.commandEncoder,
    frame.output,
    blur.pipelines.maskComposite,
    [
      sourceGroup,
      blur.blurLevels[0].group,
      maskGroupFor(blur, mask),
      blur.compositeParamsGroup,
    ],
  );
}

// ---------------------------------------------------------------------------
// Background image.
// ---------------------------------------------------------------------------

function imageTransform(
  outputWidth: number,
  outputHeight: number,
  imageWidth: number,
  imageHeight: number,
  fit: "cover" | "contain",
) {
  "worklet";
  const outputAspect = outputWidth / outputHeight;
  const imageAspect = imageWidth / imageHeight;
  if (fit === "cover") {
    if (imageAspect > outputAspect) {
      const scaleX = outputAspect / imageAspect;
      return {
        scaleX,
        scaleY: 1,
        offsetX: (1 - scaleX) / 2,
        offsetY: 0,
        contains: false,
      };
    }
    const scaleY = imageAspect / outputAspect;
    return {
      scaleX: 1,
      scaleY,
      offsetX: 0,
      offsetY: (1 - scaleY) / 2,
      contains: false,
    };
  }
  if (imageAspect > outputAspect) {
    const scaleY = outputAspect / imageAspect;
    return {
      scaleX: 1,
      scaleY: 1 / scaleY,
      offsetX: 0,
      offsetY: (1 - 1 / scaleY) / 2,
      contains: true,
    };
  }
  const scaleX = imageAspect / outputAspect;
  return {
    scaleX: 1 / scaleX,
    scaleY: 1,
    offsetX: (1 - 1 / scaleX) / 2,
    offsetY: 0,
    contains: true,
  };
}

function encodeImageFrame(
  state: VideoEffectKernelState,
  frame: VideoEffectFrame,
  options: BackgroundImageFrameOptions,
): void {
  "worklet";
  const image = state as ImageState;
  const mask =
    options.enabled === false
      ? null
      : currentMask(image.segmentation, frame.timestampUs);
  if (mask == null || image.image == null || image.imageGroup == null) {
    encodeCopy(image, frame);
    return;
  }
  const fit = imageTransform(
    image.width,
    image.height,
    image.image.width,
    image.image.height,
    options.fit ?? "cover",
  );
  const transform = mask.sourceUvToMaskUv;
  const backgroundColor = options.backgroundColor ?? [0, 0, 0, 1];
  image.device.queue.writeBuffer(
    image.imageParams,
    0,
    new Float32Array([
      transform[0] ?? 1,
      transform[1] ?? 0,
      transform[2] ?? 0,
      0,
      transform[3] ?? 0,
      transform[4] ?? 1,
      transform[5] ?? 0,
      0,
      fit.scaleX,
      fit.scaleY,
      fit.offsetX,
      fit.offsetY,
      clamp(options.edgeFeather, 0, 0.5, 0.08),
      fit.contains ? 1 : 0,
      0,
      0,
      backgroundColor[0],
      backgroundColor[1],
      backgroundColor[2],
      backgroundColor[3],
    ]),
  );
  drawFullscreen(
    frame.commandEncoder,
    frame.output,
    image.pipelines.imageComposite,
    [
      sourceGroupFor(image, frame.source),
      image.imageGroup,
      maskGroupFor(image, mask),
      image.imageParamsGroup,
    ],
  );
}

// ---------------------------------------------------------------------------
// Session construction (JS thread).
// ---------------------------------------------------------------------------

function createCommonState(
  context: VideoEffectContext,
): Omit<CommonState, "effectId"> {
  const pipelines = createBackgroundPipelines(
    context.device,
    context.outputFormat,
  );
  const compositeParams = createUniformBuffer(
    context.device,
    48,
    "fishjam-video-effect-mask-params",
  );
  return {
    device: context.device,
    width: context.width,
    height: context.height,
    pipelines,
    compositeParams,
    compositeParamsGroup: createUniformBindGroup(
      context.device,
      pipelines.uniformLayout,
      compositeParams,
    ),
    segmentation: null,
    sourceView: null,
    sourceGroup: null,
    maskView: null,
    maskGroup: null,
  };
}

/**
 * Prepares the segmentation provider and attaches its frame kernel to the state. A failure is
 * reported through `onStatus` and leaves the effect as a passthrough, so the caller still gets
 * a usable session.
 */
async function prepareSegmentation(
  state: CommonState,
  context: VideoEffectContext,
  provider: PersonSegmentationProvider,
): Promise<PersonSegmentationSession | null> {
  context.onStatus?.("loading");
  try {
    const session = await provider.prepare({
      device: context.device,
      outputWidth: context.width,
      outputHeight: context.height,
      onStatus: context.onStatus,
    });
    state.segmentation = session.frameKernel;
    context.onStatus?.("ready");
    return session;
  } catch (cause) {
    context.onStatus?.("error", asError(cause));
    return null;
  }
}

function disposeCommon(
  state: CommonState,
  segmentation: PersonSegmentationSession | null,
): void {
  segmentation?.dispose();
  state.compositeParams.destroy();
}

export async function createBackgroundBlurSession(
  context: VideoEffectContext,
  getOptions: () => BackgroundBlurOptions,
): Promise<VideoEffectSession<BackgroundBlurFrameOptions>> {
  const common = createCommonState(context);
  const blurLevels: BlurLevel[] = [];
  for (let index = 0; index < MAX_BLUR_LEVELS; index += 1) {
    const divisor = 2 ** (index + 1);
    const width = Math.max(1, Math.floor(context.width / divisor));
    const height = Math.max(1, Math.floor(context.height / divisor));
    const { texture, view } = createSampleTexture(
      context.device,
      width,
      height,
      `fishjam-video-effect-blur-level-${index}`,
    );
    blurLevels.push({
      texture,
      view,
      width,
      height,
      group: createTextureBindGroup(
        context.device,
        common.pipelines.textureLayout,
        view,
        common.pipelines.sampler,
      ),
    });
  }
  const state: BlurState = {
    ...common,
    effectId: "fishjam.background-blur",
    blurLevels,
  };
  const segmentation = await prepareSegmentation(
    state,
    context,
    getOptions().segmentation,
  );
  const frameKernel: VideoEffectFrameKernel<BackgroundBlurFrameOptions> = {
    state,
    offer: offerToSegmentation,
    encode: encodeBlurFrame,
    reset: resetSegmentation,
  };
  // Disposal is tracked here, not in `state`: the state may already have been copied to a
  // worklet runtime, and mutating it afterwards neither reaches that copy nor is allowed.
  let disposed = false;
  return {
    frameKernel,
    offer: (input) => {
      if (!disposed) offerToSegmentation(state, input);
    },
    encode: (frame) => {
      if (!disposed) encodeBlurFrame(state, frame, getOptions());
    },
    reset: () => {
      if (!disposed) resetSegmentation(state);
    },
    dispose: () => {
      if (disposed) return;
      disposed = true;
      disposeCommon(state, segmentation);
      for (const level of state.blurLevels) level.texture.destroy();
    },
  };
}

export async function createBackgroundImageSession(
  context: VideoEffectContext,
  getOptions: () => BackgroundImageOptions,
): Promise<VideoEffectSession<BackgroundImageFrameOptions>> {
  const imageParams = createUniformBuffer(
    context.device,
    80,
    "fishjam-video-effect-image-params",
  );
  const common = createCommonState(context);
  const state: ImageState = {
    ...common,
    effectId: "fishjam.background-image",
    imageParams,
    imageParamsGroup: createUniformBindGroup(
      context.device,
      common.pipelines.uniformLayout,
      imageParams,
    ),
    image: null,
    imageGroup: null,
  };

  // The image is decoded on the JS thread. Replacing it later updates this state in place,
  // which a worklet runtime holding its own copy will not observe.
  let imageSource: BackgroundImageOptions["image"] | null = null;
  let loadingImage: Promise<void> | null = null;
  const ensureImage = (
    source: BackgroundImageOptions["image"],
  ): Promise<void> => {
    if (loadingImage != null || sameImageSource(imageSource, source)) {
      return loadingImage ?? Promise.resolve();
    }
    loadingImage = loadImageTexture(context.device, source)
      .then((image) => {
        const previous = state.image;
        state.image = image;
        imageSource = source;
        state.imageGroup = createTextureBindGroup(
          context.device,
          state.pipelines.textureLayout,
          image.view,
          state.pipelines.sampler,
        );
        previous?.texture.destroy();
      })
      .catch(() => {
        // The effect remains a passthrough until the caller supplies a valid image.
      })
      .finally(() => {
        loadingImage = null;
      });
    return loadingImage;
  };

  const [segmentation] = await Promise.all([
    prepareSegmentation(state, context, getOptions().segmentation),
    ensureImage(getOptions().image),
  ]);
  const frameKernel: VideoEffectFrameKernel<BackgroundImageFrameOptions> = {
    state,
    offer: offerToSegmentation,
    encode: encodeImageFrame,
    reset: resetSegmentation,
  };
  let disposed = false;
  return {
    frameKernel,
    offer: (input) => {
      if (!disposed) offerToSegmentation(state, input);
    },
    encode: (frame) => {
      if (disposed) return;
      const options = getOptions();
      if (!sameImageSource(imageSource, options.image)) {
        void ensureImage(options.image);
      }
      encodeImageFrame(state, frame, options);
    },
    reset: () => {
      if (!disposed) resetSegmentation(state);
    },
    dispose: () => {
      if (disposed) return;
      disposed = true;
      disposeCommon(state, segmentation);
      state.imageParams.destroy();
      state.image?.texture.destroy();
      state.image = null;
    },
  };
}

function sameImageSource(
  left: BackgroundImageOptions["image"] | null,
  right: BackgroundImageOptions["image"],
): boolean {
  return (
    left === right ||
    (left?.uri === right.uri &&
      left?.data === right.data &&
      left?.mimeType === right.mimeType)
  );
}

function asError(cause: unknown): Error {
  return cause instanceof Error ? cause : new Error(String(cause));
}

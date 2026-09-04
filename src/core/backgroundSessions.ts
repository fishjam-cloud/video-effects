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
  BackgroundBlurOptions,
  BackgroundImageOptions,
  PersonMask,
  PersonSegmentationSession,
  SegmentationInput,
  VideoEffectContext,
  VideoEffectFrame,
  VideoEffectSession,
} from "./types";

type SharedOptions = BackgroundBlurOptions | BackgroundImageOptions;

interface CommonResources {
  readonly pipelines: BackgroundPipelines;
  readonly compositeParams: GPUBuffer;
  readonly compositeParamsGroup: GPUBindGroup;
  sourceView?: GPUTextureView;
  sourceGroup?: GPUBindGroup;
  maskView?: GPUTextureView;
  maskGroup?: GPUBindGroup;
}

function currentMask(
  session: PersonSegmentationSession | null,
  timestampUs: number,
): PersonMask | null {
  const mask = session?.latest(timestampUs) ?? null;
  return mask != null && timestampUs - mask.timestampUs <= MASK_MAX_AGE_US
    ? mask
    : null;
}

function writeMaskParams(
  device: GPUDevice,
  buffer: GPUBuffer,
  mask: PersonMask,
  edgeFeather: number,
): void {
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
      0,
      0,
      0,
    ]),
  );
}

abstract class BaseBackgroundSession<
  Options extends SharedOptions,
> implements VideoEffectSession {
  protected readonly device: GPUDevice;
  protected readonly width: number;
  protected readonly height: number;
  protected readonly getOptions: () => Options;
  protected readonly common: CommonResources;
  protected segmentation: PersonSegmentationSession | null = null;
  private disposed = false;

  constructor(context: VideoEffectContext, getOptions: () => Options) {
    this.device = context.device;
    this.width = context.width;
    this.height = context.height;
    this.getOptions = getOptions;
    const pipelines = createBackgroundPipelines(
      context.device,
      context.outputFormat,
    );
    const compositeParams = createUniformBuffer(
      context.device,
      48,
      "fishjam-video-effect-mask-params",
    );
    this.common = {
      pipelines,
      compositeParams,
      compositeParamsGroup: createUniformBindGroup(
        context.device,
        pipelines.uniformLayout,
        compositeParams,
      ),
    };
  }

  async initialize(context: VideoEffectContext): Promise<void> {
    context.onStatus?.("loading");
    try {
      this.segmentation = await this.getOptions().segmentation.prepare({
        device: context.device,
        outputWidth: context.width,
        outputHeight: context.height,
        onStatus: context.onStatus,
      });
      context.onStatus?.("ready");
    } catch (cause) {
      context.onStatus?.("error", asError(cause));
    }
  }

  offer(input: SegmentationInput): void {
    if (!this.disposed) this.segmentation?.offer(input);
  }

  encode(frame: VideoEffectFrame): void {
    if (this.disposed) return;
    const options = this.getOptions();
    const mask =
      options.enabled === false
        ? null
        : currentMask(this.segmentation, frame.timestampUs);
    if (mask == null) {
      this.encodeCopy(frame);
      return;
    }
    this.encodeEffect(frame, mask, options);
  }

  reset(): void {
    this.segmentation?.reset();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.segmentation?.dispose();
    this.segmentation = null;
    this.common.compositeParams.destroy();
  }

  protected sourceGroup(source: GPUTextureView): GPUBindGroup {
    if (this.common.sourceView !== source) {
      this.common.sourceView = source;
      this.common.sourceGroup = createTextureBindGroup(
        this.device,
        this.common.pipelines.textureLayout,
        source,
        this.common.pipelines.sampler,
      );
    }
    return this.common.sourceGroup!;
  }

  protected maskGroup(mask: PersonMask): GPUBindGroup {
    if (this.common.maskView !== mask.texture) {
      this.common.maskView = mask.texture;
      this.common.maskGroup = createTextureBindGroup(
        this.device,
        this.common.pipelines.textureLayout,
        mask.texture,
        this.common.pipelines.sampler,
      );
    }
    return this.common.maskGroup!;
  }

  protected encodeCopy(frame: VideoEffectFrame): void {
    drawFullscreen(
      frame.commandEncoder,
      frame.output,
      this.common.pipelines.copy,
      [this.sourceGroup(frame.source)],
    );
  }

  protected writeCommonMaskParams(
    mask: PersonMask,
    options: SharedOptions,
  ): void {
    writeMaskParams(
      this.device,
      this.common.compositeParams,
      mask,
      clamp(options.edgeFeather, 0, 0.5, 0.08),
    );
  }

  protected abstract encodeEffect(
    frame: VideoEffectFrame,
    mask: PersonMask,
    options: Options,
  ): void;
}

export class BackgroundBlurSession extends BaseBackgroundSession<BackgroundBlurOptions> {
  private readonly blurX: { texture: GPUTexture; view: GPUTextureView };
  private readonly blurY: { texture: GPUTexture; view: GPUTextureView };
  private readonly horizontalBlurParams: GPUBuffer;
  private readonly verticalBlurParams: GPUBuffer;
  private readonly horizontalBlurParamsGroup: GPUBindGroup;
  private readonly verticalBlurParamsGroup: GPUBindGroup;
  private blurXGroup: GPUBindGroup | null = null;

  constructor(
    context: VideoEffectContext,
    getOptions: () => BackgroundBlurOptions,
  ) {
    super(context, getOptions);
    const width = Math.max(1, Math.ceil(context.width / 4));
    const height = Math.max(1, Math.ceil(context.height / 4));
    this.blurX = createSampleTexture(
      context.device,
      width,
      height,
      "fishjam-video-effect-blur-x",
    );
    this.blurY = createSampleTexture(
      context.device,
      width,
      height,
      "fishjam-video-effect-blur-y",
    );
    this.horizontalBlurParams = createUniformBuffer(
      context.device,
      16,
      "fishjam-video-effect-blur-horizontal-params",
    );
    this.verticalBlurParams = createUniformBuffer(
      context.device,
      16,
      "fishjam-video-effect-blur-vertical-params",
    );
    this.horizontalBlurParamsGroup = createUniformBindGroup(
      context.device,
      this.common.pipelines.uniformLayout,
      this.horizontalBlurParams,
    );
    this.verticalBlurParamsGroup = createUniformBindGroup(
      context.device,
      this.common.pipelines.uniformLayout,
      this.verticalBlurParams,
    );
  }

  protected encodeEffect(
    frame: VideoEffectFrame,
    mask: PersonMask,
    options: BackgroundBlurOptions,
  ): void {
    const radius = clamp(options.radius, 0, 40, 18);
    if (radius === 0) {
      this.encodeCopy(frame);
      return;
    }
    const sourceGroup = this.sourceGroup(frame.source);
    this.device.queue.writeBuffer(
      this.horizontalBlurParams,
      0,
      new Float32Array([1, 0, radius * 0.2, 0]),
    );
    drawFullscreen(
      frame.commandEncoder,
      this.blurX.view,
      this.common.pipelines.blur,
      [sourceGroup, this.horizontalBlurParamsGroup],
    );

    this.blurXGroup ??= createTextureBindGroup(
      this.device,
      this.common.pipelines.textureLayout,
      this.blurX.view,
      this.common.pipelines.sampler,
    );
    this.device.queue.writeBuffer(
      this.verticalBlurParams,
      0,
      new Float32Array([0, 1, radius * 0.25, 0]),
    );
    drawFullscreen(
      frame.commandEncoder,
      this.blurY.view,
      this.common.pipelines.blur,
      [this.blurXGroup, this.verticalBlurParamsGroup],
    );

    this.writeCommonMaskParams(mask, options);
    const blurredGroup = createTextureBindGroup(
      this.device,
      this.common.pipelines.textureLayout,
      this.blurY.view,
      this.common.pipelines.sampler,
    );
    drawFullscreen(
      frame.commandEncoder,
      frame.output,
      this.common.pipelines.maskComposite,
      [
        sourceGroup,
        blurredGroup,
        this.maskGroup(mask),
        this.common.compositeParamsGroup,
      ],
    );
  }

  override dispose(): void {
    super.dispose();
    this.horizontalBlurParams.destroy();
    this.verticalBlurParams.destroy();
    this.blurX.texture.destroy();
    this.blurY.texture.destroy();
  }
}

export class BackgroundImageSession extends BaseBackgroundSession<BackgroundImageOptions> {
  private readonly imageParams: GPUBuffer;
  private readonly imageParamsGroup: GPUBindGroup;
  private image: LoadedImageTexture | null = null;
  private imageGroup: GPUBindGroup | null = null;
  private imageSource: BackgroundImageOptions["image"] | null = null;
  private loadingImage: Promise<void> | null = null;

  constructor(
    context: VideoEffectContext,
    getOptions: () => BackgroundImageOptions,
  ) {
    super(context, getOptions);
    this.imageParams = createUniformBuffer(
      context.device,
      80,
      "fishjam-video-effect-image-params",
    );
    this.imageParamsGroup = createUniformBindGroup(
      context.device,
      this.common.pipelines.uniformLayout,
      this.imageParams,
    );
    void this.ensureImage(getOptions().image);
  }

  protected encodeEffect(
    frame: VideoEffectFrame,
    mask: PersonMask,
    options: BackgroundImageOptions,
  ): void {
    if (!sameImageSource(this.imageSource, options.image))
      void this.ensureImage(options.image);
    if (this.image == null || this.imageGroup == null) {
      this.encodeCopy(frame);
      return;
    }
    const fit = imageTransform(
      this.width,
      this.height,
      this.image.width,
      this.image.height,
      options.fit ?? "cover",
    );
    const transform = mask.sourceUvToMaskUv;
    const backgroundColor = options.backgroundColor ?? [0, 0, 0, 1];
    this.device.queue.writeBuffer(
      this.imageParams,
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
      this.common.pipelines.imageComposite,
      [
        this.sourceGroup(frame.source),
        this.imageGroup,
        this.maskGroup(mask),
        this.imageParamsGroup,
      ],
    );
  }

  override dispose(): void {
    super.dispose();
    this.imageParams.destroy();
    this.image?.texture.destroy();
    this.image = null;
  }

  private async ensureImage(
    source: BackgroundImageOptions["image"],
  ): Promise<void> {
    if (this.loadingImage != null || sameImageSource(this.imageSource, source))
      return;
    this.loadingImage = loadImageTexture(this.device, source)
      .then((image) => {
        const previous = this.image;
        this.image = image;
        this.imageSource = source;
        this.imageGroup = createTextureBindGroup(
          this.device,
          this.common.pipelines.textureLayout,
          image.view,
          this.common.pipelines.sampler,
        );
        previous?.texture.destroy();
      })
      .catch(() => {
        // The effect remains a passthrough until the caller supplies a valid image.
      })
      .finally(() => {
        this.loadingImage = null;
      });
    await this.loadingImage;
  }
}

function imageTransform(
  outputWidth: number,
  outputHeight: number,
  imageWidth: number,
  imageHeight: number,
  fit: "cover" | "contain",
) {
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

export async function createBackgroundBlurSession(
  context: VideoEffectContext,
  getOptions: () => BackgroundBlurOptions,
): Promise<VideoEffectSession> {
  const session = new BackgroundBlurSession(context, getOptions);
  await session.initialize(context);
  return session;
}

export async function createBackgroundImageSession(
  context: VideoEffectContext,
  getOptions: () => BackgroundImageOptions,
): Promise<VideoEffectSession> {
  const session = new BackgroundImageSession(context, getOptions);
  await session.initialize(context);
  return session;
}

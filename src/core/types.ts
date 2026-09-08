/// <reference types="@webgpu/types" preserve="true" />

export type VideoEffectStatus =
  "idle" | "loading" | "ready" | "unsupported" | "error";

export type SegmentationInputKind =
  "web-frame" | "native-rgb-frame" | "gpu-texture";

export interface SegmentationContext {
  readonly device: GPUDevice;
  readonly outputWidth: number;
  readonly outputHeight: number;
  readonly onStatus?: (status: VideoEffectStatus, error?: Error) => void;
}

export interface GpuSegmentationInput {
  readonly kind: "gpu-texture";
  readonly timestampUs: number;
  /** Dimensions of {@link texture}, in pixels. */
  readonly width: number;
  readonly height: number;
  /**
   * The upright RGBA frame the effect composites this frame — the same view passed as
   * {@link VideoEffectFrame.source} — so the mask lines up with the output by construction.
   */
  readonly texture: GPUTextureView;
  readonly commandEncoder: GPUCommandEncoder;
}

export interface NativeRgbSegmentationInput {
  readonly kind: "native-rgb-frame";
  readonly timestampUs: number;
  readonly width: number;
  readonly height: number;
  /** Owned by the source adapter and valid only for this callback. */
  readonly frame: unknown;
}

export interface WebFrameSegmentationInput {
  readonly kind: "web-frame";
  readonly timestampUs: number;
  readonly width: number;
  readonly height: number;
  readonly frame: VideoFrame | ImageBitmap;
}

export type SegmentationInput =
  GpuSegmentationInput | NativeRgbSegmentationInput | WebFrameSegmentationInput;

export interface PersonMask {
  readonly texture: GPUTextureView;
  readonly timestampUs: number;
  /** Affine 3×3 matrix mapping source UV into mask UV. */
  readonly sourceUvToMaskUv: Float32Array;
}

/**
 * Plain-data state of a segmentation session. It holds only numbers, strings and GPU objects,
 * so a worklet runtime can copy it; pass it back into every {@link PersonSegmentationFrameKernel}
 * call.
 */
export interface PersonSegmentationKernelState {
  readonly providerId: string;
}

/**
 * The per-frame entry points of a segmentation session as standalone worklet functions.
 * Use this instead of the session methods when frames are processed on another JS runtime
 * (for example a camera thread): capture the kernel once and call its functions with
 * `kernel.state`.
 */
export interface PersonSegmentationFrameKernel {
  readonly state: PersonSegmentationKernelState;
  offer(state: PersonSegmentationKernelState, input: SegmentationInput): void;
  latest(
    state: PersonSegmentationKernelState,
    renderTimestampUs: number,
  ): PersonMask | null;
  reset(state: PersonSegmentationKernelState): void;
}

export interface PersonSegmentationSession {
  offer(frame: SegmentationInput): void;
  latest(renderTimestampUs: number): PersonMask | null;
  reset(): void;
  dispose(): void;
  readonly frameKernel: PersonSegmentationFrameKernel;
}

export interface PersonSegmentationProvider {
  readonly id: string;
  readonly input: SegmentationInputKind;
  prepare(context: SegmentationContext): Promise<PersonSegmentationSession>;
}

export interface VideoEffectContext {
  readonly device: GPUDevice;
  readonly width: number;
  readonly height: number;
  readonly outputFormat: GPUTextureFormat;
  readonly onStatus?: (status: VideoEffectStatus, error?: Error) => void;
}

export interface VideoEffectFrame {
  readonly timestampUs: number;
  /** A persistent, upright RGBA source texture view supplied by the source adapter. */
  readonly source: GPUTextureView;
  readonly output: GPUTextureView;
  readonly commandEncoder: GPUCommandEncoder;
}

/**
 * Plain-data state of an effect session; see {@link PersonSegmentationKernelState}.
 */
export interface VideoEffectKernelState {
  readonly effectId: string;
}

/**
 * The per-frame entry points of an effect session as standalone worklet functions, for callers
 * that encode frames on another JS runtime. `FrameOptions` are the effect's visual options,
 * supplied on every call because a copied state cannot observe later changes.
 */
export interface VideoEffectFrameKernel<FrameOptions = unknown> {
  readonly state: VideoEffectKernelState;
  offer(state: VideoEffectKernelState, input: SegmentationInput): void;
  encode(
    state: VideoEffectKernelState,
    frame: VideoEffectFrame,
    options: FrameOptions,
  ): void;
  reset(state: VideoEffectKernelState): void;
}

export interface VideoEffectSession<FrameOptions = unknown> {
  encode(frame: VideoEffectFrame): void;
  offer(input: SegmentationInput): void;
  reset(): void;
  dispose(): void;
  readonly frameKernel: VideoEffectFrameKernel<FrameOptions>;
}

export interface VideoEffect<FrameOptions = unknown> {
  readonly id: string;
  readonly segmentationInput: SegmentationInputKind;
  create(
    context: VideoEffectContext,
  ): Promise<VideoEffectSession<FrameOptions>>;
}

export interface SegmentationOptions {
  readonly segmentation: PersonSegmentationProvider;
  readonly enabled?: boolean;
  readonly edgeFeather?: number;
}

export interface BackgroundBlurOptions extends SegmentationOptions {
  readonly radius?: number;
}

/** The blur options read on every frame: everything except the segmentation provider. */
export type BackgroundBlurFrameOptions = Omit<
  BackgroundBlurOptions,
  "segmentation"
>;

export interface VideoEffectImageSource {
  readonly uri?: string;
  readonly data?: ArrayBuffer | ArrayBufferView;
  readonly mimeType?: string;
}

export interface BackgroundImageOptions extends SegmentationOptions {
  readonly image: VideoEffectImageSource;
  readonly fit?: "cover" | "contain";
  readonly backgroundColor?: readonly [number, number, number, number];
}

/**
 * The image-background options read on every frame: everything except the segmentation
 * provider and the image itself, which is decoded once on the JS thread.
 */
export type BackgroundImageFrameOptions = Omit<
  BackgroundImageOptions,
  "segmentation" | "image"
>;

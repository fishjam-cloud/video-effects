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
  readonly width: number;
  readonly height: number;
  readonly texture: GPUTextureView;
  readonly externalTexture?: GPUExternalTexture;
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

export interface PersonSegmentationSession {
  offer(frame: SegmentationInput): void;
  latest(renderTimestampUs: number): PersonMask | null;
  reset(): void;
  dispose(): void;
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
  readonly externalTexture?: GPUExternalTexture;
}

export interface VideoEffectSession {
  encode(frame: VideoEffectFrame): void;
  offer(input: SegmentationInput): void;
  reset(): void;
  dispose(): void;
}

export interface VideoEffect {
  readonly id: string;
  readonly segmentationInput: SegmentationInputKind;
  create(context: VideoEffectContext): Promise<VideoEffectSession>;
}

export interface SegmentationOptions {
  readonly segmentation: PersonSegmentationProvider;
  readonly enabled?: boolean;
  readonly edgeFeather?: number;
}

export interface BackgroundBlurOptions extends SegmentationOptions {
  readonly radius?: number;
}

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

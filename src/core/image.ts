/// <reference types="@webgpu/types" preserve="true" />

import { TEXTURE_USAGE } from "./constants";
import type { VideoEffectImageSource } from "./types";

export interface LoadedImageTexture {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  readonly width: number;
  readonly height: number;
}

function asArrayBuffer(source: ArrayBuffer | ArrayBufferView): ArrayBuffer {
  if (source instanceof ArrayBuffer) return source;
  return source.buffer.slice(
    source.byteOffset,
    source.byteOffset + source.byteLength,
  ) as ArrayBuffer;
}

async function decodeImage(
  source: VideoEffectImageSource,
): Promise<ImageBitmap> {
  const bytes =
    source.data != null
      ? asArrayBuffer(source.data)
      : await (await fetchRequired(source.uri)).arrayBuffer();
  const createBitmap = globalThis.createImageBitmap as unknown as (
    value: Blob | ArrayBuffer,
  ) => Promise<ImageBitmap>;
  const blob = toBlob(bytes, source.mimeType);
  return createBitmap(blob ?? bytes);
}

// Browsers decode a Blob. React Native has a Blob global too, but it cannot wrap an
// ArrayBuffer (it throws), and react-native-webgpu's createImageBitmap takes the bytes directly.
function toBlob(bytes: ArrayBuffer, mimeType: string | undefined): Blob | null {
  if (typeof Blob === "undefined") return null;
  try {
    return new Blob([bytes], { type: mimeType ?? "image/*" });
  } catch {
    return null;
  }
}

function fetchRequired(uri: string | undefined): Promise<Response> {
  if (uri == null)
    return Promise.reject(
      new Error("Background image requires either uri or data."),
    );
  return fetch(uri);
}

export async function loadImageTexture(
  device: GPUDevice,
  source: VideoEffectImageSource,
): Promise<LoadedImageTexture> {
  const bitmap = await decodeImage(source);
  try {
    const texture = device.createTexture({
      label: "fishjam-video-effect-background-image",
      format: "rgba8unorm",
      size: [bitmap.width, bitmap.height],
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    });
    device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [
      bitmap.width,
      bitmap.height,
    ]);
    return {
      texture,
      view: texture.createView(),
      width: bitmap.width,
      height: bitmap.height,
    };
  } finally {
    bitmap.close?.();
  }
}

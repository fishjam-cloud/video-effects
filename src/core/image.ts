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
  if (typeof Blob !== "undefined") {
    return createBitmap(
      new Blob([bytes], { type: source.mimeType ?? "image/*" }),
    );
  }
  return createBitmap(bytes);
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

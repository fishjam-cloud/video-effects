// @ts-nocheck
/**
 * Worklet-safe byte packers for the two per-frame uniform structs.
 *
 * The TypeGPU example writes these uniforms with `buffer.write(crop)` on the JS
 * thread every frame. We compute the crop in the VisionCamera worklet, so we
 * pack the bytes by hand (tgpu's `d.struct` schema objects carry functions and
 * cannot be captured into a worklet) and upload them with
 * `device.queue.writeBuffer(...)`.
 *
 * Byte layouts verified against tgpu 0.11.x `writeToArrayBuffer`:
 *
 *   FrameCropParams (40 bytes):
 *     [0]   u32 sourceSize.x      [1]   u32 sourceSize.y
 *     [2]   f32 cropOrigin.x      [3]   f32 cropOrigin.y
 *     [4]   f32 cropSize.x        [5]   f32 cropSize.y
 *     [6..9] f32 uvTransform mat2x2f, column-major (m00, m01, m10, m11)
 *
 *   UpsampleParams (48 bytes): FrameCropParams (40) + [10] u32 edgeAware + 1 pad.
 *
 * `mat2x2f(a, b, c, d)` stores column-major columns `(a, b)` and `(c, d)`.
 * The orientation matrices use the same scalar order, so
 * we forward the four scalars unchanged.
 */

export interface FrameCrop {
  sourceWidth: number;
  sourceHeight: number;
  cropOriginX: number;
  cropOriginY: number;
  cropSizeX: number;
  cropSizeY: number;
  // mat2x2f scalars in the same (m00, m01, m10, m11) order the reference uses.
  uv00: number;
  uv01: number;
  uv10: number;
  uv11: number;
}

const FRAME_CROP_BYTES = 40;
const UPSAMPLE_BYTES = 48;

/** Packs FrameCropParams (used by the preprocess kernel + the composite). */
export function packFrameCropParams(crop: FrameCrop): ArrayBuffer {
  "worklet";
  const buffer = new ArrayBuffer(FRAME_CROP_BYTES);
  const u32 = new Uint32Array(buffer);
  const f32 = new Float32Array(buffer);
  u32[0] = crop.sourceWidth >>> 0;
  u32[1] = crop.sourceHeight >>> 0;
  f32[2] = crop.cropOriginX;
  f32[3] = crop.cropOriginY;
  f32[4] = crop.cropSizeX;
  f32[5] = crop.cropSizeY;
  f32[6] = crop.uv00;
  f32[7] = crop.uv01;
  f32[8] = crop.uv10;
  f32[9] = crop.uv11;
  return buffer;
}

/** Packs UpsampleParams (FrameCropParams + edgeAware flag). */
export function packUpsampleParams(
  crop: FrameCrop,
  edgeAware: boolean,
): ArrayBuffer {
  "worklet";
  const buffer = new ArrayBuffer(UPSAMPLE_BYTES);
  const u32 = new Uint32Array(buffer);
  const f32 = new Float32Array(buffer);
  u32[0] = crop.sourceWidth >>> 0;
  u32[1] = crop.sourceHeight >>> 0;
  f32[2] = crop.cropOriginX;
  f32[3] = crop.cropOriginY;
  f32[4] = crop.cropSizeX;
  f32[5] = crop.cropSizeY;
  f32[6] = crop.uv00;
  f32[7] = crop.uv01;
  f32[8] = crop.uv10;
  f32[9] = crop.uv11;
  u32[10] = edgeAware ? 1 : 0;
  return buffer;
}

/**
 * Square center-crop of a (sourceWidth x sourceHeight) frame, plus the chosen
 * orientation transform — worklet port of the example's `squareCrop`.
 */
export function computeSquareCrop(
  sourceWidth: number,
  sourceHeight: number,
  uv00: number,
  uv01: number,
  uv10: number,
  uv11: number,
): FrameCrop {
  "worklet";
  const size = Math.min(sourceWidth, sourceHeight);
  return {
    sourceWidth,
    sourceHeight,
    cropOriginX: Math.floor((sourceWidth - size) / 2),
    cropOriginY: Math.floor((sourceHeight - size) / 2),
    cropSizeX: size,
    cropSizeY: size,
    uv00,
    uv01,
    uv10,
    uv11,
  };
}

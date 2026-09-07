export const MASK_MAX_AGE_US = 250_000;

export const TEXTURE_USAGE = {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08,
  RENDER_ATTACHMENT: 0x10,
} as const;

export const BUFFER_USAGE = {
  COPY_DST: 0x08,
  UNIFORM: 0x40,
} as const;

export const IDENTITY_UV_TRANSFORM = new Float32Array([
  1, 0, 0, 0, 1, 0, 0, 0, 1,
]);

export function clamp(
  value: number | undefined,
  minimum: number,
  maximum: number,
  fallback: number,
): number {
  "worklet";
  if (value == null || !Number.isFinite(value)) return fallback;
  return Math.min(maximum, Math.max(minimum, value));
}

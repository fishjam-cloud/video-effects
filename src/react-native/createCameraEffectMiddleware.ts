/// <reference types="@webgpu/types" />
import type { TrackMiddleware } from "@fishjam-cloud/react-native-client";

import type { VideoEffect, VideoEffectStatus } from "../core/types";
import {
  type CameraFrameInfo,
  createCameraFrameProcessorSession,
} from "./webgpu/cameraFrameProcessorSession";
import {
  createCameraTextureResolver,
  resolveCameraTexture,
} from "./webgpu/cameraTextureResolver";
import type { WebGpuFrameRenderFunction } from "./webgpu/frameRenderContext";
import { getOutputSurfaceFormat } from "./webgpu/requiredFeatures";
import { getCameraWebGpuDevice } from "./webgpu/useCameraWebGpuDevice";

const DEFAULT_WIDTH = 720;
const DEFAULT_HEIGHT = 1280;

export interface CameraEffectMiddlewareOptions {
  /** Width of the published video, in pixels. Defaults to 720. */
  readonly width?: number;
  /** Height of the published video, in pixels. Defaults to 1280. */
  readonly height?: number;
  /** Reports the effect's loading progress and failures. */
  readonly onStatus?: (status: VideoEffectStatus, error?: Error) => void;
}

/**
 * A camera-track middleware that applies an effect to the camera track Fishjam publishes. Hand it
 * to `useCamera().setCameraTrackMiddleware`; it stays in place across screens until it is
 * replaced or cleared with `null`, and `currentCameraMiddleware` tells whether it is active.
 *
 * ```ts
 * const blur = createCameraEffectMiddleware(createBackgroundBlurEffect(() => ({ segmentation })));
 * setCameraTrackMiddleware(enabled ? blur : null);
 * ```
 */
export function createCameraEffectMiddleware(
  effect: VideoEffect,
  options: CameraEffectMiddlewareOptions = {},
): TrackMiddleware {
  const width = options.width ?? DEFAULT_WIDTH;
  const height = options.height ?? DEFAULT_HEIGHT;

  return async (rawTrack) => {
    const device = await getCameraWebGpuDevice();
    const session = await effect.create({
      device,
      width,
      height,
      outputFormat: getOutputSurfaceFormat(),
      onStatus: options.onStatus,
    });
    const resolvedCamera = createCameraTextureResolver(device, {
      width,
      height,
      cameraPixelLayout: "rgb",
    });
    const kernel = session.frameKernel;
    const frameOptions = effect.frameOptions?.();

    const frameKernel = (
      frame: CameraFrameInfo,
      render: WebGpuFrameRenderFunction,
    ) => {
      "worklet";
      render((context) => {
        "worklet";
        const timestampUs = Math.floor(frame.timestampNanoseconds / 1_000);
        resolveCameraTexture(
          context.device,
          resolvedCamera,
          context.cameraTexture,
          context.cameraWidth,
          context.cameraHeight,
          context.commandEncoder,
        );
        kernel.offer(kernel.state, {
          kind: "gpu-texture",
          timestampUs,
          width: resolvedCamera.width,
          height: resolvedCamera.height,
          texture: resolvedCamera.view,
          commandEncoder: context.commandEncoder,
        });
        kernel.encode(
          kernel.state,
          {
            timestampUs,
            source: resolvedCamera.view,
            output: context.outputView,
            commandEncoder: context.commandEncoder,
          },
          frameOptions,
        );
      });
    };

    const processor = await createCameraFrameProcessorSession({
      track: rawTrack,
      device,
      width,
      height,
      frameKernel,
    });
    return {
      track: processor.track,
      onClear: () => {
        void processor.dispose();
        session.dispose();
        resolvedCamera.texture.destroy();
      },
    };
  };
}

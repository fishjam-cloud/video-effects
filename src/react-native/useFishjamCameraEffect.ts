/// <reference types="@webgpu/types" />
import {
  type TrackMiddleware,
  useCamera,
} from "@fishjam-cloud/react-native-client";
import { useCallback, useEffect, useMemo, useState } from "react";

import type {
  VideoEffect,
  VideoEffectSession,
  VideoEffectStatus,
} from "../core/types";
import { toError } from "./internal/toError";
import {
  type CameraFrameInfo,
  createCameraFrameProcessorSession,
} from "./webgpu/cameraFrameProcessorSession";
import {
  createCameraPassthroughPipeline,
  encodeCameraPassthrough,
} from "./webgpu/cameraPassthroughPipeline";
import {
  createCameraTextureResolver,
  resolveCameraTexture,
} from "./webgpu/cameraTextureResolver";
import { computeAspectFillCrop } from "./webgpu/cropUtilities";
import type { WebGpuFrameRenderFunction } from "./webgpu/frameRenderContext";
import { getOutputSurfaceFormat } from "./webgpu/requiredFeatures";
import { useCameraWebGpuDevice } from "./webgpu/useCameraWebGpuDevice";

const DEFAULT_WIDTH = 720;
const DEFAULT_HEIGHT = 1280;

export interface FishjamCameraEffectOptions {
  /** Width of the published video, in pixels. Defaults to 720. */
  readonly width?: number;
  /** Height of the published video, in pixels. Defaults to 1280. */
  readonly height?: number;
}

export interface FishjamCameraEffectResult {
  readonly status: VideoEffectStatus;
  readonly error: Error | null;
  /** Builds the effect again after an error. */
  readonly retry: () => void;
}

/**
 * Applies an effect to the camera track Fishjam publishes, through `useCamera`'s camera-track
 * middleware. Pass `null` to publish the plain camera again. While the effect is still loading,
 * the camera is published untouched, so the track is never black.
 *
 * ```tsx
 * const blur = useBackgroundBlur({ segmentation, radius: 24 });
 * const { status, error } = useFishjamCameraEffect(blurEnabled ? blur : null);
 * ```
 */
export function useFishjamCameraEffect(
  effect: VideoEffect | null,
  options: FishjamCameraEffectOptions = {},
): FishjamCameraEffectResult {
  const width = options.width ?? DEFAULT_WIDTH;
  const height = options.height ?? DEFAULT_HEIGHT;
  const { device, error: deviceError } = useCameraWebGpuDevice();
  const { setCameraTrackMiddleware } = useCamera();
  const [attempt, setAttempt] = useState(0);
  const [session, setSession] = useState<VideoEffectSession | null>(null);
  const [status, setStatus] = useState<VideoEffectStatus>(
    effect == null ? "idle" : "loading",
  );
  const [error, setError] = useState<Error | null>(null);

  const resolvedCamera = useMemo(
    () =>
      device == null
        ? null
        : createCameraTextureResolver(device, {
            width,
            height,
            cameraPixelLayout: "rgb",
          }),
    [device, width, height],
  );
  useEffect(() => () => resolvedCamera?.texture.destroy(), [resolvedCamera]);

  const passthrough = useMemo(
    () =>
      device == null
        ? null
        : createCameraPassthroughPipeline(device, {
            cameraPixelLayout: "rgb",
            outputFormat: getOutputSurfaceFormat(),
          }),
    [device],
  );

  useEffect(() => {
    if (effect == null) {
      setStatus("idle");
      setError(null);
      return;
    }
    if (deviceError != null) {
      setStatus("unsupported");
      setError(deviceError);
      return;
    }
    if (device == null) {
      setStatus("loading");
      setError(null);
      return;
    }
    let active = true;
    let created: VideoEffectSession | null = null;
    setStatus("loading");
    setError(null);
    void effect
      .create({
        device,
        width,
        height,
        outputFormat: getOutputSurfaceFormat(),
        onStatus: (nextStatus, nextError) => {
          if (!active) return;
          setStatus(nextStatus);
          setError(nextError ?? null);
        },
      })
      .then((newSession) => {
        created = newSession;
        if (!active) {
          newSession.dispose();
          return;
        }
        setSession(newSession);
        setStatus("ready");
      })
      .catch((cause: unknown) => {
        if (!active) return;
        setStatus("error");
        setError(toError(cause));
      });
    return () => {
      active = false;
      setSession(null);
      created?.dispose();
    };
  }, [effect, device, deviceError, width, height, attempt]);

  const frameOptions = useFrameOptions(effect);
  const outputAspect = width / height;

  const frameKernel = useCallback(
    (frame: CameraFrameInfo, render: WebGpuFrameRenderFunction) => {
      "worklet";
      render((context) => {
        "worklet";
        if (session == null || resolvedCamera == null) {
          if (passthrough == null) return;
          encodeCameraPassthrough(
            context.device,
            passthrough,
            context.cameraTexture,
            context.outputView,
            context.commandEncoder,
            computeAspectFillCrop(
              context.cameraWidth,
              context.cameraHeight,
              outputAspect,
            ),
          );
          return;
        }
        const timestampUs = Math.floor(frame.timestampNanoseconds / 1_000);
        resolveCameraTexture(
          context.device,
          resolvedCamera,
          context.cameraTexture,
          context.cameraWidth,
          context.cameraHeight,
          context.commandEncoder,
        );
        const kernel = session.frameKernel;
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
    },
    [session, resolvedCamera, passthrough, frameOptions, outputAspect],
  );

  useEffect(() => {
    if (
      effect == null ||
      device == null ||
      resolvedCamera == null ||
      passthrough == null
    ) {
      void setCameraTrackMiddleware(null);
      return;
    }
    const middleware: TrackMiddleware = async (rawTrack) => {
      const processor = await createCameraFrameProcessorSession({
        track: rawTrack,
        device,
        width,
        height,
        frameKernel,
      });
      return {
        track: processor.track,
        onClear: () => void processor.dispose(),
      };
    };
    void setCameraTrackMiddleware(middleware);
    return () => {
      void setCameraTrackMiddleware(null);
    };
  }, [
    effect,
    device,
    resolvedCamera,
    passthrough,
    frameKernel,
    width,
    height,
    setCameraTrackMiddleware,
  ]);

  const retry = useCallback(() => setAttempt((value) => value + 1), []);
  return { status, error, retry };
}

// The frame kernel runs on the camera runtime and cannot read the effect's live options, so
// they are captured as plain data and only change identity when their values do.
function useFrameOptions(effect: VideoEffect | null): unknown {
  const current = effect?.frameOptions?.() ?? null;
  const serialized = JSON.stringify(current);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => current, [serialized]);
}

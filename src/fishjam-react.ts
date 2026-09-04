import { useCamera } from "@fishjam-cloud/react-client";
import { useEffect } from "react";

import type { VideoEffect } from "./core/types";
import { useVideoTrackEffect, type VideoTrackEffectResult } from "./web";

/** Applies a web video effect through Fishjam's existing camera-track middleware. */
export function useFishjamCameraEffect(
  effect: VideoEffect | null,
): Omit<VideoTrackEffectResult, "middleware"> {
  const { setCameraTrackMiddleware } = useCamera();
  const { middleware, status, error, retry } = useVideoTrackEffect(effect);

  useEffect(() => {
    void setCameraTrackMiddleware(effect == null ? null : middleware);
    return () => {
      void setCameraTrackMiddleware(null);
    };
  }, [effect, middleware, setCameraTrackMiddleware]);

  return { status, error, retry };
}

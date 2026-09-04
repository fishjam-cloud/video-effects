import { useMemo, useRef } from "react";

import { createBackgroundImageEffect } from "./core/effects";
import type { BackgroundImageOptions, VideoEffect } from "./core/types";

export { createBackgroundImageEffect } from "./core/effects";
export type {
  BackgroundImageOptions,
  VideoEffectImageSource as BackgroundImageSource,
} from "./core/types";

/** A stable effect descriptor; the previous image remains live while a replacement loads. */
export function useBackgroundImage(
  options: BackgroundImageOptions,
): VideoEffect {
  const optionsRef = useRef(options);
  optionsRef.current = options;
  return useMemo(
    () => createBackgroundImageEffect(() => optionsRef.current),
    // Replacing the provider requires a fresh session; visual options do not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [options.segmentation],
  );
}

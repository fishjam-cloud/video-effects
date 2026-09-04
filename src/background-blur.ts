import { useMemo, useRef } from "react";

import { createBackgroundBlurEffect } from "./core/effects";
import type { BackgroundBlurOptions, VideoEffect } from "./core/types";

export { createBackgroundBlurEffect } from "./core/effects";
export type { BackgroundBlurOptions } from "./core/types";

/** A stable effect descriptor; visual option changes update without rebuilds. */
export function useBackgroundBlur(options: BackgroundBlurOptions): VideoEffect {
  const optionsRef = useRef(options);
  optionsRef.current = options;
  return useMemo(
    () => createBackgroundBlurEffect(() => optionsRef.current),
    // Replacing the provider requires a fresh session; visual options do not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [options.segmentation],
  );
}

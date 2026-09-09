# `@fishjam-cloud/video-effects`

WebGPU background effects and person-segmentation providers for Fishjam video sources.

## Installation

```sh
yarn add @fishjam-cloud/video-effects
```

React Native applications using the TypeGPU provider also need compatible WebGPU and Worklets integrations. The provider requires React Native Worklets 0.12.1 or newer and the New Architecture.

## Background blur

```tsx
import { useBackgroundBlur } from "@fishjam-cloud/video-effects/background-blur";
import { typeGpuPersonSegmentation } from "@fishjam-cloud/video-effects/segmentation/typegpu";

const segmentation = typeGpuPersonSegmentation();
const effect = useBackgroundBlur({ segmentation, radius: 18 });
```

Pass `effect` to a compatible Fishjam video-source adapter. The TypeGPU provider consumes GPU-texture input and ships with the model in `assets/selfie_segmenter.ssgbin`.

## Background image

```tsx
import { useBackgroundImage } from "@fishjam-cloud/video-effects/background-image";

const effect = useBackgroundImage({
  segmentation,
  image: { uri: "https://example.com/background.jpg" },
  fit: "cover",
});
```

## Fishjam React Native camera

`@fishjam-cloud/video-effects/fishjam-react-native` runs an effect on the camera track Fishjam already publishes, through `useCamera`'s camera-track middleware. No separate camera and no custom track: the app keeps using `useCamera`, and remote peers keep seeing `peer.cameraTrack`.

```tsx
import { useBackgroundBlur } from "@fishjam-cloud/video-effects/background-blur";
import { useFishjamCameraEffect } from "@fishjam-cloud/video-effects/fishjam-react-native";
import { typeGpuPersonSegmentation } from "@fishjam-cloud/video-effects/segmentation/typegpu";

const segmentation = typeGpuPersonSegmentation({ modelUrl });

function BackgroundBlur({ enabled }: { enabled: boolean }) {
  const blur = useBackgroundBlur({ segmentation, radius: 24 });
  const { status, error } = useFishjamCameraEffect(enabled ? blur : null);
  return null;
}
```

Mount it anywhere inside `FishjamProvider`. While the effect loads, the plain camera is published, so the track is never black. `status` goes `loading` → `ready`, or `unsupported` / `error` with `error` set; `retry()` builds the effect again.

The model file ships in the package: `require("@fishjam-cloud/video-effects/assets/selfie_segmenter.ssgbin")` through `expo-asset` gives the `modelUrl`. Add `ssgbin` to Metro's `resolver.assetExts`.

The app must have these installed and linked: `@fishjam-cloud/react-native-client`, `@fishjam-cloud/react-native-webrtc` (0.30.2 or newer), `@fishjam-cloud/react-native-worklets`, `react-native-worklets` (0.12 or newer, with its Babel plugin) and `react-native-webgpu`, with the New Architecture on. Android needs API 26.

For custom rendering, `createCameraFrameProcessorSession` and the WebGPU helpers are exported from the same entry; the hook is built on them.

## Entry points

- `@fishjam-cloud/video-effects` — provider and effect contracts
- `@fishjam-cloud/video-effects/background-blur` — blur factory and React hook
- `@fishjam-cloud/video-effects/background-image` — image-replacement factory and React hook
- `@fishjam-cloud/video-effects/segmentation/typegpu` — experimental GPU-only segmentation provider
- `@fishjam-cloud/video-effects/web` — browser track middleware
- `@fishjam-cloud/video-effects/fishjam-react` — Fishjam React adapter
- `@fishjam-cloud/video-effects/fishjam-react-native` — Fishjam React Native camera-track session

Effects improve presentation; they are not privacy or security boundaries.

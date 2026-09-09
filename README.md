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

`@fishjam-cloud/video-effects/fishjam-react-native` runs an effect on the camera track Fishjam already publishes. Hand it the raw camera track from `useCamera`'s `setCameraTrackMiddleware`, draw each frame in a worklet, and return `session.track` from the middleware. No separate camera or custom track is needed.

```tsx
import { useCamera } from "@fishjam-cloud/react-native-client";
import {
  createCameraFrameProcessorSession,
  useCameraWebGpuDevice,
} from "@fishjam-cloud/video-effects/fishjam-react-native";

const { device } = useCameraWebGpuDevice();
const { setCameraTrackMiddleware } = useCamera();

setCameraTrackMiddleware(async (rawTrack) => {
  const session = await createCameraFrameProcessorSession({
    track: rawTrack,
    device,
    width: 720,
    height: 1280,
    frameKernel,
  });
  return { track: session.track, onClear: () => void session.dispose() };
});
```

The app must have these installed and linked: `@fishjam-cloud/react-native-webrtc` (0.30.2 or newer), `react-native-webgpu`, `react-native-worklets` and `@fishjam-cloud/react-native-worklets`.

## Entry points

- `@fishjam-cloud/video-effects` — provider and effect contracts
- `@fishjam-cloud/video-effects/background-blur` — blur factory and React hook
- `@fishjam-cloud/video-effects/background-image` — image-replacement factory and React hook
- `@fishjam-cloud/video-effects/segmentation/typegpu` — experimental GPU-only segmentation provider
- `@fishjam-cloud/video-effects/web` — browser track middleware
- `@fishjam-cloud/video-effects/fishjam-react` — Fishjam React adapter
- `@fishjam-cloud/video-effects/fishjam-react-native` — Fishjam React Native camera-track session

Effects improve presentation; they are not privacy or security boundaries.

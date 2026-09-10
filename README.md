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

```ts
import { createBackgroundBlurEffect } from "@fishjam-cloud/video-effects/background-blur";
import { createCameraEffectMiddleware } from "@fishjam-cloud/video-effects/fishjam-react-native";
import { typeGpuPersonSegmentation } from "@fishjam-cloud/video-effects/segmentation/typegpu";

const segmentation = typeGpuPersonSegmentation({ modelUrl });
export const backgroundBlur = createCameraEffectMiddleware(
  createBackgroundBlurEffect(() => ({ segmentation, radius: 24 })),
);
```

```tsx
const { currentCameraMiddleware, setCameraTrackMiddleware } = useCamera();
const isBlurOn = currentCameraMiddleware === backgroundBlur;
<Button
  onPress={() => setCameraTrackMiddleware(isBlurOn ? null : backgroundBlur)}
/>;
```

The middleware lives in Fishjam's camera state, so it stays applied across screens until it is cleared with `null`. Switch it on once the camera is on. Pass `onStatus` in the options to follow loading and errors.

For a component-scoped version there is `useFishjamCameraEffect(effect | null)`, the twin of the web hook; it clears the effect when the component unmounts.

The model file ships in the package: `require("@fishjam-cloud/video-effects/assets/selfie_segmenter.ssgbin")` through `expo-asset` gives the `modelUrl`. Add `ssgbin` to Metro's `resolver.assetExts`.

The app must have these installed and linked: `@fishjam-cloud/react-native-client`, `@fishjam-cloud/react-native-webrtc` (0.30.2 or newer), `@fishjam-cloud/react-native-worklets`, `react-native-worklets` (0.12 or newer, with its Babel plugin) and `react-native-webgpu`, with the New Architecture on. Android needs API 26.

## Entry points

- `@fishjam-cloud/video-effects` — provider and effect contracts
- `@fishjam-cloud/video-effects/background-blur` — blur factory and React hook
- `@fishjam-cloud/video-effects/background-image` — image-replacement factory and React hook
- `@fishjam-cloud/video-effects/segmentation/typegpu` — experimental GPU-only segmentation provider
- `@fishjam-cloud/video-effects/web` — browser track middleware
- `@fishjam-cloud/video-effects/fishjam-react` — Fishjam React adapter
- `@fishjam-cloud/video-effects/fishjam-react-native` — Fishjam React Native camera-track session

Effects improve presentation; they are not privacy or security boundaries.

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

## Entry points

- `@fishjam-cloud/video-effects` — provider and effect contracts
- `@fishjam-cloud/video-effects/background-blur` — blur factory and React hook
- `@fishjam-cloud/video-effects/background-image` — image-replacement factory and React hook
- `@fishjam-cloud/video-effects/segmentation/typegpu` — experimental GPU-only segmentation provider
- `@fishjam-cloud/video-effects/web` — browser track middleware
- `@fishjam-cloud/video-effects/fishjam-react` — Fishjam React adapter

Effects improve presentation; they are not privacy or security boundaries.

# Changelog

## 0.1.3

- `createCameraEffectMiddleware(effect)` for React Native: a camera-track middleware for `useCamera().setCameraTrackMiddleware`, so an effect can be switched on from any screen without a provider or a hook.
- The TypeGPU segmentation model is fetched and parsed once per URL and shared by later sessions, so switching an effect on again or restarting the camera does not reload it.
- `typeGpuPersonSegmentation({ loadModel })` supplies the model bytes for platforms that cannot `fetch` the model URL, such as an asset embedded in an Android release build.

## 0.1.2

- Background blur: the person no longer bleeds into the blurred background (no halo), the outline is a smooth ramp that hugs the body, and similar-coloured background next to the person is no longer left sharp.
- The outline weight is computed once per frame, so the frame's GPU cost is unchanged from 0.1.1.

## 0.1.1

- Web: the camera effect renders again (the resolve pass targeted the canvas format), feeds the segmentation its GPU texture so background blur applies, and the packaged default model path resolves.
- A segmentation load failure is no longer hidden behind a ready status.

## 0.1.0

- `useFishjamCameraEffect` for React Native: one hook applies an effect to the Fishjam camera track and publishes the plain camera while the effect loads.
- Effects run on Fishjam's own camera track through the camera middleware: new `fishjam-react-native` entry with `createCameraFrameProcessorSession` and `useCameraWebGpuDevice`.
- Worklet-safe frame kernels, dual Kawase background blur, TypeGPU 0.12 support.
- Segmentation and compositing cover the whole resolved frame.

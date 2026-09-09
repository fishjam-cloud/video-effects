# Changelog

## 0.1.0

- `useFishjamCameraEffect` for React Native: one hook applies an effect to the Fishjam camera track and publishes the plain camera while the effect loads.
- Effects run on Fishjam's own camera track through the camera middleware: new `fishjam-react-native` entry with `createCameraFrameProcessorSession` and `useCameraWebGpuDevice`.
- Worklet-safe frame kernels, dual Kawase background blur, TypeGPU 0.12 support.
- Segmentation and compositing cover the whole resolved frame.

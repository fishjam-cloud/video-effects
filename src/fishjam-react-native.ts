/// <reference types="@webgpu/types" />
/**
 * Runs video effects on Fishjam's own React Native camera track. Hand the raw camera track from
 * `useCamera`'s middleware to {@link createCameraFrameProcessorSession}; it renders your frame
 * kernel on the GPU and republishes the result as the peer's camera track.
 *
 * Requires `@fishjam-cloud/react-native-webrtc`, `react-native-webgpu`, `react-native-worklets` and
 * `@fishjam-cloud/react-native-webrtc-worklets` in the app.
 *
 * @packageDocumentation
 */

export {
  type CameraFrameInfo,
  type CameraFrameKernel,
  type CameraFrameProcessorSession,
  createCameraFrameProcessorSession,
  type CreateCameraFrameProcessorSessionOptions,
} from "./react-native/webgpu/cameraFrameProcessorSession";
export {
  type CameraPassthroughPipeline,
  type CameraPassthroughPipelineOptions,
  createCameraPassthroughPipeline,
  encodeCameraPassthrough,
} from "./react-native/webgpu/cameraPassthroughPipeline";
export {
  type CameraPixelLayout,
  type CameraShaderBindings,
  createCameraBindGroup,
  createCameraShaderBindings,
  type CreateCameraShaderBindingsOptions,
  type SampleCameraFn,
} from "./react-native/webgpu/cameraShaderBindings";
export {
  type CameraTextureResolver,
  createCameraTextureResolver,
  type CreateCameraTextureResolverOptions,
  resolveCameraTexture,
} from "./react-native/webgpu/cameraTextureResolver";
export {
  computeAspectFillCrop,
  type FrameCrop,
} from "./react-native/webgpu/cropUtilities";
export type {
  WebGpuFrameRenderContext,
  WebGpuFrameRenderFunction,
} from "./react-native/webgpu/frameRenderContext";
export {
  getOutputSurfaceFormat,
  getRequiredWebGpuCameraFeatures,
} from "./react-native/webgpu/requiredFeatures";
export {
  useCameraWebGpuDevice,
  type UseCameraWebGpuDeviceResult,
} from "./react-native/webgpu/useCameraWebGpuDevice";

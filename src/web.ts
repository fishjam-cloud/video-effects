/// <reference types="@webgpu/types" preserve="true" />

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type {
  VideoEffect,
  VideoEffectSession,
  VideoEffectStatus,
} from "./core/types";

// The resolve pass renders the camera into this texture, so its pipeline must target this format,
// not the canvas format (BGRA on macOS), or every frame is rejected and the track stays black.
const SOURCE_TEXTURE_FORMAT: GPUTextureFormat = "rgba8unorm";

export type VideoTrackMiddleware = (track: MediaStreamTrack) => Promise<{
  track: MediaStreamTrack;
  onClear: () => void;
}>;

export interface VideoTrackEffectResult {
  readonly middleware: VideoTrackMiddleware;
  readonly status: VideoEffectStatus;
  readonly error: Error | null;
  readonly retry: () => void;
}

/**
 * Creates a browser track middleware around a provider-neutral effect. The
 * source track remains owned by the caller; only the generated canvas track is
 * stopped during cleanup.
 */
export function useVideoTrackEffect(
  effect: VideoEffect | null,
): VideoTrackEffectResult {
  const [attempt, setAttempt] = useState(0);
  const [status, setStatus] = useState<VideoEffectStatus>(
    effect == null ? "idle" : "loading",
  );
  const [error, setError] = useState<Error | null>(null);
  const rendererRef = useRef<WebTrackRenderer | null>(null);

  useEffect(() => {
    setStatus(effect == null ? "idle" : "loading");
    setError(null);
  }, [effect, attempt]);

  useEffect(() => () => rendererRef.current?.dispose(), []);

  const middleware = useMemo<VideoTrackMiddleware>(() => {
    if (effect == null) {
      return async (track) => ({ track, onClear: () => undefined });
    }
    return async (track) => {
      rendererRef.current?.dispose();
      const renderer = await WebTrackRenderer.create(
        track,
        effect,
        (nextStatus, nextError) => {
          setStatus(nextStatus);
          setError(nextError ?? null);
        },
      );
      rendererRef.current = renderer;
      return { track: renderer.track, onClear: () => renderer.dispose() };
    };
    // A retry deliberately creates a new middleware identity for the camera adapter.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effect, attempt]);

  const retry = useCallback(() => setAttempt((value) => value + 1), []);
  return { middleware, status, error, retry };
}

class WebTrackRenderer {
  readonly track: MediaStreamTrack;
  private readonly video: HTMLVideoElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: GPUCanvasContext;
  private readonly device: GPUDevice;
  private readonly effect: VideoEffect;
  private readonly publishTrack: CanvasCaptureMediaStreamTrack;
  private readonly sourceTexture: GPUTexture;
  private readonly source: GPUTextureView;
  private session: VideoEffectSession | null = null;
  private resolvePipeline: GPURenderPipeline;
  private frameCallbackId: number | null = null;
  private disposed = false;

  private constructor(
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    device: GPUDevice,
    effect: VideoEffect,
    publishTrack: CanvasCaptureMediaStreamTrack,
    resolvePipeline: GPURenderPipeline,
    sourceTexture: GPUTexture,
  ) {
    this.video = video;
    this.canvas = canvas;
    this.context = context;
    this.device = device;
    this.effect = effect;
    this.publishTrack = publishTrack;
    this.track = publishTrack;
    this.resolvePipeline = resolvePipeline;
    this.sourceTexture = sourceTexture;
    this.source = sourceTexture.createView();
  }

  static async create(
    inputTrack: MediaStreamTrack,
    effect: VideoEffect,
    onStatus: (status: VideoEffectStatus, error?: Error) => void,
  ): Promise<WebTrackRenderer> {
    if (!("gpu" in navigator)) {
      onStatus("unsupported");
      throw new Error("WebGPU is unavailable in this browser.");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter == null) {
      onStatus("unsupported");
      throw new Error("No compatible WebGPU adapter is available.");
    }
    const device = await adapter.requestDevice();
    const video = document.createElement("video");
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    video.srcObject = new MediaStream([inputTrack]);
    video.style.cssText =
      "position:fixed;width:1px;height:1px;opacity:0;pointer-events:none;";
    document.body.appendChild(video);
    await video.play();
    await waitForVideoDimensions(video);

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("webgpu");
    if (context == null)
      throw new Error("Could not create a WebGPU canvas context.");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });
    const stream = canvas.captureStream(0);
    const publishTrack = stream.getVideoTracks()[0] as
      CanvasCaptureMediaStreamTrack | undefined;
    if (publishTrack == null)
      throw new Error("Could not capture the WebGPU canvas.");
    const sourceTexture = device.createTexture({
      label: "fishjam-web-effect-source",
      size: [canvas.width, canvas.height],
      format: SOURCE_TEXTURE_FORMAT,
      usage:
        GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const renderer = new WebTrackRenderer(
      video,
      canvas,
      context,
      device,
      effect,
      publishTrack,
      createExternalResolvePipeline(device, SOURCE_TEXTURE_FORMAT),
      sourceTexture,
    );
    try {
      renderer.session = await effect.create({
        device,
        width: canvas.width,
        height: canvas.height,
        outputFormat: format,
        onStatus,
      });
      renderer.scheduleNextFrame();
      return renderer;
    } catch (cause) {
      renderer.dispose();
      const error = cause instanceof Error ? cause : new Error(String(cause));
      onStatus("error", error);
      throw error;
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.frameCallbackId != null)
      this.video.cancelVideoFrameCallback(this.frameCallbackId);
    this.session?.dispose();
    this.session = null;
    this.sourceTexture.destroy();
    this.publishTrack.stop();
    this.video.pause();
    this.video.srcObject = null;
    this.video.remove();
  }

  private scheduleNextFrame(): void {
    this.frameCallbackId = this.video.requestVideoFrameCallback(
      (_, metadata) => {
        this.render(metadata.mediaTime * 1_000_000);
        if (!this.disposed) this.scheduleNextFrame();
      },
    );
  }

  private render(timestampUs: number): void {
    if (this.disposed || this.session == null) return;
    const externalTexture = this.device.importExternalTexture({
      source: this.video,
    });
    const output = this.context.getCurrentTexture().createView();
    const encoder = this.device.createCommandEncoder();
    const resolveGroup = this.device.createBindGroup({
      layout: this.resolvePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: externalTexture }],
    });
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view: this.source, loadOp: "clear", storeOp: "store" },
      ],
    });
    pass.setPipeline(this.resolvePipeline);
    pass.setBindGroup(0, resolveGroup);
    pass.draw(3);
    pass.end();
    if (this.effect.segmentationInput === "gpu-texture") {
      this.session.offer({
        kind: "gpu-texture",
        timestampUs,
        width: this.canvas.width,
        height: this.canvas.height,
        texture: this.source,
        commandEncoder: encoder,
      });
    } else if (this.effect.segmentationInput === "web-frame") {
      void createImageBitmap(this.video).then((frame) =>
        this.session?.offer({
          kind: "web-frame",
          frame,
          timestampUs,
          width: frame.width,
          height: frame.height,
        }),
      );
    }
    this.session.encode({
      timestampUs,
      source: this.source,
      output,
      commandEncoder: encoder,
    });
    this.device.queue.submit([encoder.finish()]);
    this.publishTrack.requestFrame();
  }
}

function createExternalResolvePipeline(
  device: GPUDevice,
  format: GPUTextureFormat,
): GPURenderPipeline {
  return device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({
        code: `@vertex fn main(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.,1.); }`,
      }),
      entryPoint: "main",
    },
    fragment: {
      module: device.createShaderModule({
        code: `@group(0) @binding(0) var source:texture_external; @fragment fn main(@builtin(position) p:vec4f)->@location(0) vec4f { return textureLoad(source, vec2i(p.xy)); }`,
      }),
      entryPoint: "main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });
}

function waitForVideoDimensions(video: HTMLVideoElement): Promise<void> {
  if (video.videoWidth > 0 && video.videoHeight > 0) return Promise.resolve();
  return new Promise((resolve) =>
    video.addEventListener("loadedmetadata", () => resolve(), { once: true }),
  );
}

// @ts-nocheck
/**
 * Flattens TypeGPU dispatch state into raw WebGPU objects. TypeGPU class instances
 * cannot cross a worklets runtime boundary, while WebGPU host objects can.
 * Frame-dependent external-texture groups are rebuilt by the caller every frame.
 */
import type {
  TgpuBindGroup,
  TgpuBindGroupLayout,
  TgpuComputePipeline,
  TgpuRoot,
} from "typegpu";

import type { KernelHandle } from "./inference/kernels/types";

// A static (non-per-frame) bind group as raw natives + its group index.
export interface RecordedStaticGroup {
  index: number;
  bindGroup: GPUBindGroup;
}

// One compute dispatch flattened to raw natives, ready to replay in the worklet.
// `frameGroupIndex` is set ONLY for the preprocess + upsample dispatches: it is
// the group index the worklet must bind a freshly-built external-texture group
// to (everything else is already in `staticGroups`).
export interface RecordedComputeDispatch {
  pipeline: GPUComputePipeline;
  staticGroups: RecordedStaticGroup[];
  workgroupsX: number;
  workgroupsY: number;
  frameGroupIndex: number; // -1 when the dispatch has no external-texture group
}

function groupIndexOf(layout: TgpuBindGroupLayout): number {
  if (layout.index == null) {
    throw new Error(
      "recordComputeDispatch: every bind group layout must pin its group index with `$idx(...)`, so the replayed dispatch binds each group where the shader expects it.",
    );
  }
  return layout.index;
}

/**
 * Records one compute dispatch as raw natives. `staticBindGroups` are the
 * dispatch's non-frame groups; `frameLayout`, when present, is the
 * external-texture group the worklet binds per frame. Group indices come from
 * the layouts' pinned `$idx`, which TypeGPU honours when compiling the pipeline,
 * so nothing has to be replayed to discover them.
 */
export function recordComputeDispatch(
  root: TgpuRoot,
  handle: Pick<KernelHandle, "pipeline"> & {
    pipeline: TgpuComputePipeline;
  },
  staticBindGroups: TgpuBindGroup[],
  workgroups: { x: number; y?: number },
  options?: { frameLayout?: TgpuBindGroupLayout },
): RecordedComputeDispatch {
  const pipeline = root.unwrap(handle.pipeline) as GPUComputePipeline;
  const staticGroups = staticBindGroups.map((group) => ({
    index: groupIndexOf(group.layout),
    bindGroup: root.unwrap(group),
  }));
  return {
    pipeline,
    staticGroups,
    workgroupsX: workgroups.x,
    workgroupsY: workgroups.y ?? 1,
    frameGroupIndex: options?.frameLayout
      ? groupIndexOf(options.frameLayout)
      : -1,
  };
}

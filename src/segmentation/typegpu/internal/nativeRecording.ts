// @ts-nocheck
/**
 * Records TypeGPU dispatch state as raw WebGPU objects. TypeGPU class instances
 * cannot cross a Worklets runtime boundary, while WebGPU host objects can.
 * Frame-dependent external-texture groups are rebuilt by the caller.
 */
import type {
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

// A recording proxy that satisfies the subset of GPUComputePassEncoder that
// tgpu's `_applyComputeState` + `dispatchWorkgroups` touch.
interface RecordingState {
  pipeline: GPUComputePipeline | null;
  groups: RecordedStaticGroup[];
}

function makeRecordingPass(state: RecordingState): GPUComputePassEncoder {
  // Only setPipeline / setBindGroup are exercised by tgpu's apply step. We throw
  // on anything unexpected so a tgpu internals change surfaces loudly at setup
  // time (on the JS thread) instead of silently producing a wrong frame.
  const proxy = {
    setPipeline(pipeline: GPUComputePipeline) {
      state.pipeline = pipeline;
    },
    setBindGroup(index: number, bindGroup: GPUBindGroup) {
      state.groups.push({ index, bindGroup });
    },
    // tgpu may call these; they are no-ops for recording.
    pushDebugGroup() {},
    popDebugGroup() {},
    insertDebugMarker() {},
  } as unknown as GPUComputePassEncoder;
  return proxy;
}

/**
 * Records a tgpu compute dispatch into raw natives by replaying it against a
 * proxy pass. `staticBindGroups` are the dispatch's NON-frame bind groups (every
 * group for fully-static dispatches; all-but-the-frame-group for the two
 * external-texture dispatches). `frameLayout`, when present, lets us learn the
 * frame group's index without binding it.
 */
export function recordComputeDispatch(
  root: TgpuRoot,
  handle: Pick<KernelHandle, "pipeline"> & {
    pipeline: TgpuComputePipeline;
  },
  staticBindGroups: KernelHandle["bindGroup"][],
  workgroups: { x: number; y?: number },
  options?: { frameLayout?: TgpuBindGroupLayout; expectedGroupCount?: number },
): RecordedComputeDispatch {
  const state: RecordingState = { pipeline: null, groups: [] };
  const proxyPass = makeRecordingPass(state);

  // Build the tgpu dispatch chain bound to the proxy pass + the static groups,
  // then dispatch — tgpu issues setPipeline + setBindGroup(index, rawGroup) for
  // each, which the proxy records. (Identical call shape to the example's
  // `handle.pipeline.with(pass).with(handle.bindGroup).dispatchWorkgroups(n)`.)
  let chain = handle.pipeline.with(proxyPass);
  for (const group of staticBindGroups) {
    chain = chain.with(group);
  }
  // For a frame-dependent dispatch the external-texture group is deliberately
  // NOT supplied, so tgpu's `applyBindGroups` records every static group via the
  // proxy and THEN throws `MissingBindGroupsError`. That throw is expected — we
  // already have what we need (setPipeline + the static setBindGroups). For
  // fully-static dispatches no throw should occur, so we only swallow when a
  // frame layout is expected.
  try {
    chain.dispatchWorkgroups(workgroups.x, workgroups.y ?? 1);
  } catch (error) {
    if (!options?.frameLayout) {
      throw error;
    }
  }

  if (!state.pipeline) {
    throw new Error(
      "recordComputeDispatch: tgpu did not call setPipeline (internals changed?)",
    );
  }

  let frameGroupIndex = -1;
  if (options?.frameLayout) {
    // The frame group is the single index in [0, total) not covered by the
    // recorded static groups. `expectedGroupCount` must equal the pipeline's
    // total bind-group count (static groups + 1 frame group).
    const total = options.expectedGroupCount ?? state.groups.length + 1;
    const used = new Set(state.groups.map((g) => g.index));
    for (let index = 0; index < total; index++) {
      if (!used.has(index)) {
        frameGroupIndex = index;
        break;
      }
    }
    if (frameGroupIndex < 0) {
      throw new Error(
        "recordComputeDispatch: could not locate the external-texture group index",
      );
    }
  }

  return {
    pipeline: state.pipeline,
    staticGroups: state.groups,
    workgroupsX: workgroups.x,
    workgroupsY: workgroups.y ?? 1,
    frameGroupIndex,
  };
}

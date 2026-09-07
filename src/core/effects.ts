import {
  createBackgroundBlurSession,
  createBackgroundImageSession,
} from "./backgroundSessions";
import type {
  BackgroundBlurFrameOptions,
  BackgroundBlurOptions,
  BackgroundImageFrameOptions,
  BackgroundImageOptions,
  VideoEffect,
} from "./types";

/** Creates a model-free background blur descriptor. */
export function createBackgroundBlurEffect(
  getOptions: () => BackgroundBlurOptions,
): VideoEffect<BackgroundBlurFrameOptions> {
  return {
    id: "fishjam.background-blur",
    get segmentationInput() {
      return getOptions().segmentation.input;
    },
    create: (context) => createBackgroundBlurSession(context, getOptions),
  };
}

/** Creates a model-free image-background descriptor. */
export function createBackgroundImageEffect(
  getOptions: () => BackgroundImageOptions,
): VideoEffect<BackgroundImageFrameOptions> {
  return {
    id: "fishjam.background-image",
    get segmentationInput() {
      return getOptions().segmentation.input;
    },
    create: (context) => createBackgroundImageSession(context, getOptions),
  };
}

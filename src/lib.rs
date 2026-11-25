//! MediaPipe tasks, preview overlay mapping, verified model pinning, and LeptosKit runtime bridge.
//!
//! Core functionality (always enabled, zero dependencies):
//! - [`OverlayTransform`] maps upright-frame normalized landmarks, regions, and pixel
//!   boxes onto a `cover`, `contain`, or `fill` camera preview (with optional front-camera
//!   mirroring).
//! - Task data contracts ([`PoseLandmarkerResult`], [`ObjectDetectorResult`],
//!   [`GestureRecognizerResult`], [`AudioClassifierResult`], [`PoseJoint`], [`MpImage`],
//!   [`TimestampTracker`], [`FramePacer`]) mirror MediaPipe Tasks `v1.0.0`.
//!
//! Optional features:
//! - `serde`: derives `Serialize` and `Deserialize` on geometry and task types.
//! - `model-cache`: SHA-256 and exact byte-length verified [`ModelPin`] & [`ModelCache`].
//! - `native`: dynamic `libmediapipe.{so,dylib}` v1.0.0 C-API runtime loader & safe task wrappers.
//! - `leptoskit-guest` / `leptoskit-host`: `"mediapipe"` `leptoskit_bridge` runtime plugin.
//!
//! MediaPipe is a trademark of Google LLC. This project is independent software
//! and is not affiliated with or endorsed by Google.
//!
//! ```
//! use mediapipe::{OverlayTransform, PreviewFit, Size};
//!
//! let transform = OverlayTransform::new(
//!     Size::new(100.0, 100.0),
//!     0,
//!     Size::new(100.0, 100.0),
//!     true,
//!     PreviewFit::Cover,
//! )?;
//! let point = transform.map_normalized(0.25, 0.75);
//! assert_eq!((point.x, point.y), (75.0, 75.0));
//! # Ok::<(), mediapipe::OverlayError>(())
//! ```

#[cfg(any(feature = "leptoskit-guest", feature = "leptoskit-host"))]
pub mod bridge;
#[cfg(feature = "model-cache")]
pub mod models;
#[cfg(feature = "native")]
pub mod native;
mod overlay;
pub mod tasks;

pub use overlay::{
    BoundingBox, NormalizedLandmark, NormalizedRect, OverlayError, OverlayTransform, Point,
    PreviewFit, Rect, Size,
};
pub use tasks::{
    AudioBuffer, AudioClassifierOptions, AudioClassifierResult, Category, ClassificationHead,
    Classifications, ClassifierOptions, Detection, FramePacer, GestureRecognizerOptions,
    GestureRecognizerResult, ImageError, ImageFormat, MpImage, NormalizedKeypoint,
    ObjectDetectorOptions, ObjectDetectorResult, PoseJoint, PoseLandmarkerOptions,
    PoseLandmarkerResult, RunningMode, TimestampTracker, VisionObservation, WorldLandmark,
};

//! Map MediaPipe vision results onto a camera preview.
//!
//! Detection runs on the upright frame: the image after the rotation passed
//! to the task. A preview widget still crops (`cover`), letterboxes
//! (`contain`), or stretches (`fill`) that frame, and a front camera is
//! usually mirrored. [`OverlayTransform`] folds those three facts into one
//! mapping from normalized landmarks, normalized regions, and pixel boxes
//! to view coordinates.
//!
//! Coordinates are not rotated again here. Doing so would double-apply the
//! rotation the task already consumed.
//!
//! MediaPipe is a trademark of Google LLC. This crate is independent
//! software and is not affiliated with or endorsed by Google.
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

mod overlay;

pub use overlay::{
    BoundingBox, NormalizedLandmark, NormalizedRect, OverlayError, OverlayTransform, Point,
    PreviewFit, Rect, Size,
};

//! MediaPipe task contracts, pixel conversion, and frame pacing.
//!
//! These pure-Rust types mirror MediaPipe Tasks `v1.0.0` (pose landmarker,
//! object detector, gesture recognizer, and audio classifier) and are shared
//! across Wasm guests, mobile/desktop native hosts, and backend replay.

use std::fmt;

use crate::overlay::{BoundingBox, NormalizedLandmark, Size};

/// Running mode for vision tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RunningMode {
    /// Single independent image frames.
    Image,
    /// Decoded video frames with monotonically increasing timestamps.
    Video,
    /// Asynchronous live camera stream frames.
    LiveStream,
}

impl RunningMode {
    /// Wire / C-API numeric discriminant matching `MpRunningMode`.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::Image => 1,
            Self::Video => 2,
            Self::LiveStream => 3,
        }
    }

    /// Reconstruct from wire / C-API discriminant.
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(Self::Image),
            2 => Some(Self::Video),
            3 => Some(Self::LiveStream),
            _ => None,
        }
    }
}

/// Supported CPU image formats for MediaPipe vision tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImageFormat {
    /// 3 bytes per pixel (`R, G, B`).
    Srgb,
    /// 4 bytes per pixel (`R, G, B, A`).
    Srgba,
    /// 1 byte per pixel luminance.
    Gray8,
}

impl ImageFormat {
    /// Bytes per pixel.
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Srgb => 3,
            Self::Srgba => 4,
            Self::Gray8 => 1,
        }
    }

    /// MediaPipe C-API `MpImageFormat` value.
    #[must_use]
    pub const fn c_format(self) -> u32 {
        match self {
            Self::Srgb => 1,
            Self::Srgba => 2,
            Self::Gray8 => 3,
        }
    }
}

/// Why an image buffer or conversion failed validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    /// Image dimensions must be non-zero.
    ZeroDimensions { width: u32, height: u32 },
    /// Pixel buffer length did not match the expected byte count.
    BufferLengthMismatch { expected: usize, actual: usize },
    /// Rotation must be a multiple of 90 degrees.
    RotationNotMultipleOf90 { degrees: i32 },
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimensions { width, height } => {
                write!(f, "image dimensions must be positive, got {width}x{height}")
            }
            Self::BufferLengthMismatch { expected, actual } => {
                write!(f, "expected {expected} image bytes, got {actual}")
            }
            Self::RotationNotMultipleOf90 { degrees } => {
                write!(
                    f,
                    "rotation must be a multiple of 90 degrees, got {degrees}"
                )
            }
        }
    }
}

impl std::error::Error for ImageError {}

/// Contiguous 8-bit image frame passed into MediaPipe vision tasks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MpImage {
    format: ImageFormat,
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl MpImage {
    /// Build a validated contiguous image buffer.
    ///
    /// # Errors
    ///
    /// Fails when `width` or `height` is zero or `data.len()` does not equal
    /// `width * height * format.channels()`.
    pub fn new(
        format: ImageFormat,
        width: u32,
        height: u32,
        data: Vec<u8>,
    ) -> Result<Self, ImageError> {
        if width == 0 || height == 0 {
            return Err(ImageError::ZeroDimensions { width, height });
        }
        let expected = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(format.channels());
        if data.len() != expected {
            return Err(ImageError::BufferLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            format,
            width,
            height,
            data,
        })
    }

    /// Construct an `Srgb` image from `R, G, B` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError`] when dimensions or buffer length are invalid.
    pub fn from_srgb(width: u32, height: u32, data: Vec<u8>) -> Result<Self, ImageError> {
        Self::new(ImageFormat::Srgb, width, height, data)
    }

    /// Construct an `Srgba` image from `R, G, B, A` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError`] when dimensions or buffer length are invalid.
    pub fn from_srgba(width: u32, height: u32, data: Vec<u8>) -> Result<Self, ImageError> {
        Self::new(ImageFormat::Srgba, width, height, data)
    }

    /// Convert an iOS/macOS `BGRA8888` frame to `Srgba`.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError`] when dimensions or buffer length are invalid.
    pub fn from_bgra8(width: u32, height: u32, bgra: &[u8]) -> Result<Self, ImageError> {
        if width == 0 || height == 0 {
            return Err(ImageError::ZeroDimensions { width, height });
        }
        let expected = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        if bgra.len() != expected {
            return Err(ImageError::BufferLengthMismatch {
                expected,
                actual: bgra.len(),
            });
        }
        let mut rgba = Vec::with_capacity(expected);
        let (chunks, _) = bgra.as_chunks::<4>();
        for px in chunks {
            rgba.push(px[2]);
            rgba.push(px[1]);
            rgba.push(px[0]);
            rgba.push(px[3]);
        }
        Ok(Self {
            format: ImageFormat::Srgba,
            width,
            height,
            data: rgba,
        })
    }

    /// Convert an Android CameraX `NV21` (`Y` plane followed by interleaved `V, U`) frame to `Srgb`.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError`] when dimensions are zero or the buffer is too short.
    pub fn from_nv21(width: u32, height: u32, nv21: &[u8]) -> Result<Self, ImageError> {
        if width == 0 || height == 0 {
            return Err(ImageError::ZeroDimensions { width, height });
        }
        let w = width as usize;
        let h = height as usize;
        let y_len = w.saturating_mul(h);
        let uv_rows = h.div_ceil(2);
        let uv_row_bytes = w.div_ceil(2).saturating_mul(2);
        let expected = y_len.saturating_add(uv_rows.saturating_mul(uv_row_bytes));
        if nv21.len() < expected {
            return Err(ImageError::BufferLengthMismatch {
                expected,
                actual: nv21.len(),
            });
        }
        let mut rgb = Vec::with_capacity(y_len.saturating_mul(3));
        for row in 0..h {
            let y_row = row * w;
            let uv_row = y_len + (row / 2) * uv_row_bytes;
            for col in 0..w {
                let y = i32::from(nv21[y_row + col]);
                let uv_idx = uv_row + (col / 2) * 2;
                let v = i32::from(nv21[uv_idx]) - 128;
                let u = i32::from(nv21[uv_idx + 1]) - 128;
                let (r, g, b) = yuv_to_rgb(y, u, v);
                rgb.push(r);
                rgb.push(g);
                rgb.push(b);
            }
        }
        Ok(Self {
            format: ImageFormat::Srgb,
            width,
            height,
            data: rgb,
        })
    }

    /// Rotate the image clockwise by a multiple of 90 degrees so the output is upright.
    ///
    /// # Errors
    ///
    /// Fails when `rotation_degrees` is not a multiple of 90.
    pub fn rotate_clockwise(&self, rotation_degrees: i32) -> Result<Self, ImageError> {
        if rotation_degrees % 90 != 0 {
            return Err(ImageError::RotationNotMultipleOf90 {
                degrees: rotation_degrees,
            });
        }
        let turns = rotation_degrees.rem_euclid(360) / 90;
        if turns == 0 {
            return Ok(self.clone());
        }
        let ch = self.format.channels();
        let w = self.width as usize;
        let h = self.height as usize;
        let (out_w, out_h) = if turns % 2 == 1 { (h, w) } else { (w, h) };
        let mut out = vec![0u8; out_w * out_h * ch];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = match turns {
                    1 => (h - 1 - y, x),
                    2 => (w - 1 - x, h - 1 - y),
                    3 => (y, w - 1 - x),
                    _ => unreachable!(),
                };
                let src = (y * w + x) * ch;
                let dst = (dy * out_w + dx) * ch;
                out[dst..dst + ch].copy_from_slice(&self.data[src..src + ch]);
            }
        }
        Ok(Self {
            format: self.format,
            width: out_w as u32,
            height: out_h as u32,
            data: out,
        })
    }

    /// Pixel format.
    #[must_use]
    pub const fn format(&self) -> ImageFormat {
        self.format
    }

    /// Image width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Size as floating-point [`Size`].
    #[must_use]
    pub fn size(&self) -> Size {
        Size::new(f64::from(self.width), f64::from(self.height))
    }

    /// Contiguous pixel bytes.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

fn yuv_to_rgb(y: i32, u: i32, v: i32) -> (u8, u8, u8) {
    let c = (y - 16).max(0);
    let r = (298 * c + 409 * v + 128) >> 8;
    let g = (298 * c - 100 * u - 208 * v + 128) >> 8;
    let b = (298 * c + 516 * u + 128) >> 8;
    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

/// Floating-point PCM audio buffer for [`AudioClassifierOptions`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioBuffer {
    /// Sample rate in Hertz (for example `16_000.0` for YAMNet).
    pub sample_rate_hz: f64,
    /// Interleaved channel count (typically `1` for mono).
    pub channels: u32,
    /// Normalized PCM samples in `-1.0..=1.0`.
    pub samples: Vec<f32>,
}

/// Single category prediction from a classifier or detector.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Category {
    /// Model category index.
    pub index: i32,
    /// Confidence score in `0.0..=1.0`.
    pub score: f32,
    /// Canonical model label (for example `"sports ball"`, `"Thumb_Up"`, `"Clapping"`).
    pub category_name: Option<String>,
    /// Optional localized display label.
    pub display_name: Option<String>,
}

/// Optional normalized keypoint on an object or face detection.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NormalizedKeypoint {
    /// Horizontal normalized coordinate.
    pub x: f32,
    /// Vertical normalized coordinate.
    pub y: f32,
    /// Optional confidence score.
    pub score: Option<f32>,
}

/// Single detected object with bounding box and scored categories.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Detection {
    /// Predicted categories ordered by descending score.
    pub categories: Vec<Category>,
    /// Bounding box in upright pixel coordinates.
    pub bounding_box: BoundingBox,
    /// Optional associated keypoints.
    pub keypoints: Vec<NormalizedKeypoint>,
}

/// 3D world-space landmark in meters relative to the subject origin.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WorldLandmark {
    /// Meters along X.
    pub x: f32,
    /// Meters along Y.
    pub y: f32,
    /// Meters along Z.
    pub z: f32,
    /// Optional likelihood that the landmark is visible.
    pub visibility: Option<f32>,
    /// Optional likelihood that the landmark is present.
    pub presence: Option<f32>,
}

/// BlazePose GHUM 33 body landmark indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum PoseJoint {
    Nose = 0,
    LeftEyeInner = 1,
    LeftEye = 2,
    LeftEyeOuter = 3,
    RightEyeInner = 4,
    RightEye = 5,
    RightEyeOuter = 6,
    LeftEar = 7,
    RightEar = 8,
    MouthLeft = 9,
    MouthRight = 10,
    LeftShoulder = 11,
    RightShoulder = 12,
    LeftElbow = 13,
    RightElbow = 14,
    LeftWrist = 15,
    RightWrist = 16,
    LeftPinky = 17,
    RightPinky = 18,
    LeftIndex = 19,
    RightIndex = 20,
    LeftThumb = 21,
    RightThumb = 22,
    LeftHip = 23,
    RightHip = 24,
    LeftKnee = 25,
    RightKnee = 26,
    LeftAnkle = 27,
    RightAnkle = 28,
    LeftHeel = 29,
    RightHeel = 30,
    LeftFootIndex = 31,
    RightFootIndex = 32,
}

impl PoseJoint {
    /// Total number of BlazePose landmarks per detected pose.
    pub const COUNT: usize = 33;

    /// Zero-based landmark array index.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Look up this joint inside a normalized pose landmark slice.
    #[must_use]
    pub fn from_pose(self, landmarks: &[NormalizedLandmark]) -> Option<&NormalizedLandmark> {
        landmarks.get(self.index())
    }
}

/// Shared classifier head options (`ObjectDetector`, `GestureRecognizer`, `AudioClassifier`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClassifierOptions {
    /// Maximum results (`-1` for unbounded).
    pub max_results: i32,
    /// Minimum score threshold.
    pub score_threshold: f32,
    /// Optional category allowlist (mutually exclusive with `category_denylist`).
    pub category_allowlist: Vec<String>,
    /// Optional category denylist (mutually exclusive with `category_allowlist`).
    pub category_denylist: Vec<String>,
}

impl Default for ClassifierOptions {
    fn default() -> Self {
        Self {
            max_results: -1,
            score_threshold: 0.0,
            category_allowlist: Vec::new(),
            category_denylist: Vec::new(),
        }
    }
}

/// Configuration for `PoseLandmarker`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PoseLandmarkerOptions {
    /// Running mode (`Image`, `Video`, or `LiveStream`).
    pub running_mode: RunningMode,
    /// Maximum poses to detect.
    pub num_poses: i32,
    /// Minimum pose detection confidence.
    pub min_pose_detection_confidence: f32,
    /// Minimum pose presence confidence.
    pub min_pose_presence_confidence: f32,
    /// Minimum tracking confidence between frames.
    pub min_tracking_confidence: f32,
    /// Whether to emit segmentation masks.
    pub output_segmentation_masks: bool,
}

impl Default for PoseLandmarkerOptions {
    fn default() -> Self {
        Self {
            running_mode: RunningMode::Video,
            num_poses: 1,
            min_pose_detection_confidence: 0.5,
            min_pose_presence_confidence: 0.5,
            min_tracking_confidence: 0.5,
            output_segmentation_masks: false,
        }
    }
}

/// Output of `PoseLandmarker`.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PoseLandmarkerResult {
    /// Detected poses in normalized upright-frame coordinates.
    pub pose_landmarks: Vec<Vec<NormalizedLandmark>>,
    /// Detected poses in 3D world coordinates (meters).
    pub pose_world_landmarks: Vec<Vec<WorldLandmark>>,
}

/// Configuration for `ObjectDetector`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectDetectorOptions {
    /// Running mode (`Image`, `Video`, or `LiveStream`).
    pub running_mode: RunningMode,
    /// Classifier filtering options.
    pub classifier: ClassifierOptions,
}

impl Default for ObjectDetectorOptions {
    fn default() -> Self {
        Self {
            running_mode: RunningMode::Video,
            classifier: ClassifierOptions {
                max_results: 5,
                score_threshold: 0.2,
                ..ClassifierOptions::default()
            },
        }
    }
}

/// Output of `ObjectDetector`.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectDetectorResult {
    /// Detected objects on the upright frame.
    pub detections: Vec<Detection>,
}

/// Configuration for `GestureRecognizer`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GestureRecognizerOptions {
    /// Running mode (`Image`, `Video`, or `LiveStream`).
    pub running_mode: RunningMode,
    /// Maximum hands to detect.
    pub num_hands: i32,
    /// Minimum hand detection confidence.
    pub min_hand_detection_confidence: f32,
    /// Minimum hand presence confidence.
    pub min_hand_presence_confidence: f32,
    /// Minimum hand tracking confidence.
    pub min_tracking_confidence: f32,
    /// Filter options for canned gestures (`Thumb_Up`, `Victory`, etc.).
    pub canned_gestures: ClassifierOptions,
}

impl Default for GestureRecognizerOptions {
    fn default() -> Self {
        Self {
            running_mode: RunningMode::Video,
            num_hands: 2,
            min_hand_detection_confidence: 0.5,
            min_hand_presence_confidence: 0.5,
            min_tracking_confidence: 0.5,
            canned_gestures: ClassifierOptions::default(),
        }
    }
}

/// Output of `GestureRecognizer`.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GestureRecognizerResult {
    /// Top recognized gestures per detected hand.
    pub gestures: Vec<Vec<Category>>,
    /// Handedness classifications (`Left` / `Right`) per hand.
    pub handedness: Vec<Vec<Category>>,
    /// Normalized 21-point hand landmarks per hand.
    pub hand_landmarks: Vec<Vec<NormalizedLandmark>>,
    /// 3D world hand landmarks per hand.
    pub hand_world_landmarks: Vec<Vec<WorldLandmark>>,
}

/// Configuration for `AudioClassifier`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioClassifierOptions {
    /// Classifier filtering options.
    pub classifier: ClassifierOptions,
}

impl Default for AudioClassifierOptions {
    fn default() -> Self {
        Self {
            classifier: ClassifierOptions {
                max_results: 5,
                score_threshold: 0.0,
                ..ClassifierOptions::default()
            },
        }
    }
}

/// Predictions from one classifier head.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClassificationHead {
    /// Classifier head index.
    pub head_index: i32,
    /// Optional classifier head name.
    pub head_name: Option<String>,
    /// Predicted categories ordered by descending score.
    pub categories: Vec<Category>,
}

/// One time-windowed classification output from `AudioClassifier`.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Classifications {
    /// Optional chunk timestamp in milliseconds.
    pub timestamp_ms: Option<i64>,
    /// Output heads for this chunk.
    pub heads: Vec<ClassificationHead>,
}

/// Output of `AudioClassifier`.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioClassifierResult {
    /// Time-windowed audio classifications across the clip.
    pub results: Vec<Classifications>,
}

/// Combined frame observation emitted from a native MediaPipe session to a guest.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VisionObservation {
    /// Monotonic frame timestamp in milliseconds.
    pub timestamp_ms: i64,
    /// Upright frame width in pixels.
    pub frame_width: u32,
    /// Upright frame height in pixels.
    pub frame_height: u32,
    /// Rotation applied to produce the upright frame (multiple of 90).
    pub rotation_degrees: i32,
    /// Whether the preview comes from a mirrored front camera.
    pub mirrored: bool,
    /// Optional pose landmarker output for this frame.
    pub pose: Option<PoseLandmarkerResult>,
    /// Optional object detector output for this frame.
    pub objects: Option<ObjectDetectorResult>,
    /// Optional gesture recognizer output for this frame.
    pub gestures: Option<GestureRecognizerResult>,
}

/// Ensures timestamps passed into MediaPipe video/live-stream tasks strictly increase.
#[derive(Debug, Clone, Default)]
pub struct TimestampTracker {
    last_ms: Option<i64>,
}

impl TimestampTracker {
    /// Create a fresh tracker.
    #[must_use]
    pub const fn new() -> Self {
        Self { last_ms: None }
    }

    /// Return a strictly increasing millisecond timestamp at least `candidate_ms`.
    pub fn next(&mut self, candidate_ms: i64) -> i64 {
        let clamped = candidate_ms.max(0);
        let emitted = match self.last_ms {
            Some(prev) if clamped <= prev => prev.saturating_add(1),
            _ => clamped,
        };
        self.last_ms = Some(emitted);
        emitted
    }

    /// Reset the tracker for a new capture session.
    pub fn reset(&mut self) {
        self.last_ms = None;
    }
}

/// Drops overlapping live frames when the model runner is busy or below the frame interval.
#[derive(Debug, Clone)]
pub struct FramePacer {
    min_interval_ms: i64,
    last_accepted_ms: Option<i64>,
    in_flight: bool,
    dropped: u64,
}

impl FramePacer {
    /// Pace frames to at most one every `min_interval_ms` milliseconds.
    #[must_use]
    pub const fn new(min_interval_ms: i64) -> Self {
        Self {
            min_interval_ms: if min_interval_ms < 0 {
                0
            } else {
                min_interval_ms
            },
            last_accepted_ms: None,
            in_flight: false,
            dropped: 0,
        }
    }

    /// Returns `true` and marks a frame in flight when inference should proceed.
    pub fn try_acquire(&mut self, timestamp_ms: i64) -> bool {
        if self.in_flight {
            self.dropped = self.dropped.saturating_add(1);
            return false;
        }
        if let Some(prev) = self.last_accepted_ms
            && timestamp_ms.saturating_sub(prev) < self.min_interval_ms
        {
            self.dropped = self.dropped.saturating_add(1);
            return false;
        }
        self.in_flight = true;
        self.last_accepted_ms = Some(timestamp_ms);
        true
    }

    /// Mark the active inference frame finished.
    pub fn release(&mut self) {
        self.in_flight = false;
    }

    /// Number of frames dropped due to busyness or pacing.
    #[must_use]
    pub const fn dropped_frames(&self) -> u64 {
        self.dropped
    }
}

//! Dynamic MediaPipe v1.0.0 C-API runtime loader and safe task wrappers.
//!
//! This module binds directly to the MediaPipe Tasks `v1.0.0` C ABI exposed by
//! `libmediapipe.{so,dylib}` (the same checksummed native archives built in
//! `ifiokjr/mediapipe` release `native-v1.0.0-1`). Loading is dynamic via
//! `libloading`, so binaries compile without Bazel and fail closed with
//! [`NativeError::RuntimeUnavailable`] whenever the native shared library is
//! not installed on the host.

#![allow(unsafe_code)]

use std::ffi::{CStr, CString, c_char, c_double, c_float, c_int, c_uint, c_void};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;

use libloading::Library;

use crate::overlay::{BoundingBox, NormalizedLandmark};
use crate::tasks::{
    AudioBuffer, AudioClassifierOptions, AudioClassifierResult, Category, ClassificationHead,
    Classifications, ClassifierOptions, Detection, GestureRecognizerOptions,
    GestureRecognizerResult, MpImage, ObjectDetectorOptions, ObjectDetectorResult,
    PoseLandmarkerOptions, PoseLandmarkerResult, RunningMode, WorldLandmark,
};

/// Checksummed prebuilt native C-API archive published for MediaPipe `v1.0.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeArtifact {
    /// Target identifier (for example `"linux-x64"`, `"android-arm64"`).
    pub target: &'static str,
    /// Download URL for the immutable release archive.
    pub url: &'static str,
    /// Lowercase SHA-256 hex digest of the release zip.
    pub sha256: &'static str,
}

/// Official `native-v1.0.0-1` archive catalog matching `ifiokjr/mediapipe`.
pub const NATIVE_ARTIFACTS: &[NativeArtifact] = &[
    NativeArtifact {
        target: "android-arm64",
        url: "https://github.com/ifiokjr/mediapipe/releases/download/native-v1.0.0-1/mediapipe-v1.0.0-android-arm64.zip",
        sha256: "0a2ad71545e6908f9cf30c429d03dd4152d4495d4b860e0202a90fec9fae0fa6",
    },
    NativeArtifact {
        target: "android-x64",
        url: "https://github.com/ifiokjr/mediapipe/releases/download/native-v1.0.0-1/mediapipe-v1.0.0-android-x64.zip",
        sha256: "8636c0278a318a263e0848eeb9fab3a66bf3c657edfa721a3e469dc88531abed",
    },
    NativeArtifact {
        target: "linux-x64",
        url: "https://github.com/ifiokjr/mediapipe/releases/download/native-v1.0.0-1/mediapipe-v1.0.0-linux-x64.zip",
        sha256: "4dc233fed2d307daf7ab4d220c081bfb588c4fd794fbf0f05ebe82ef84865917",
    },
    NativeArtifact {
        target: "macos-arm64",
        url: "https://github.com/ifiokjr/mediapipe/releases/download/native-v1.0.0-1/mediapipe-v1.0.0-macos-arm64.zip",
        sha256: "49c24e9f0f32c9028cfa77da271453fb16d9bd3ad6c8d7d86a471567dfaf3c5a",
    },
];

/// Why native MediaPipe initialization or inference failed.
#[derive(Debug)]
pub enum NativeError {
    /// `libmediapipe` could not be located or loaded on this host.
    RuntimeUnavailable(String),
    /// A required C symbol was missing from the loaded library.
    MissingSymbol(String),
    /// Invalid UTF-8 or interior NUL byte in a C string parameter.
    InvalidParameter(String),
    /// MediaPipe returned a non-zero `MpStatus`.
    Status { code: u32, message: String },
}

impl fmt::Display for NativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeUnavailable(msg) => write!(f, "mediapipe runtime unavailable: {msg}"),
            Self::MissingSymbol(sym) => write!(f, "mediapipe symbol missing: {sym}"),
            Self::InvalidParameter(msg) => write!(f, "invalid mediapipe parameter: {msg}"),
            Self::Status { code, message } => {
                write!(f, "mediapipe status {code}: {message}")
            }
        }
    }
}

impl std::error::Error for NativeError {}

// --- MediaPipe v1.0.0 C ABI structures -------------------------------------

#[repr(C)]
struct MpBaseOptions {
    model_asset_buffer: *const c_char,
    model_asset_buffer_count: c_uint,
    model_asset_path: *const c_char,
    file_descriptor: c_int,
    delegate: c_uint,
    host_environment: c_uint,
    host_system: c_uint,
    host_version: *const c_char,
    ca_bundle_path: *const c_char,
    app_id: *const c_char,
    app_version: *const c_char,
}

impl Default for MpBaseOptions {
    fn default() -> Self {
        Self {
            model_asset_buffer: ptr::null(),
            model_asset_buffer_count: 0,
            model_asset_path: ptr::null(),
            file_descriptor: -1,
            delegate: 0, // CPU
            host_environment: 0,
            host_system: current_host_system(),
            host_version: ptr::null(),
            ca_bundle_path: ptr::null(),
            app_id: ptr::null(),
            app_version: ptr::null(),
        }
    }
}

const fn current_host_system() -> c_uint {
    if cfg!(target_os = "linux") {
        1
    } else if cfg!(target_os = "macos") {
        2
    } else if cfg!(target_os = "windows") {
        3
    } else if cfg!(target_os = "ios") {
        4
    } else if cfg!(target_os = "android") {
        5
    } else {
        0
    }
}

#[repr(C)]
struct MpClassifierOptions {
    display_names_locale: *const c_char,
    max_results: c_int,
    score_threshold: c_float,
    category_allowlist: *const *const c_char,
    category_allowlist_count: u32,
    category_denylist: *const *const c_char,
    category_denylist_count: u32,
}

impl Default for MpClassifierOptions {
    fn default() -> Self {
        Self {
            display_names_locale: ptr::null(),
            max_results: -1,
            score_threshold: 0.0,
            category_allowlist: ptr::null(),
            category_allowlist_count: 0,
            category_denylist: ptr::null(),
            category_denylist_count: 0,
        }
    }
}

#[repr(C)]
struct MpRect {
    left: c_int,
    top: c_int,
    bottom: c_int,
    right: c_int,
}

#[repr(C)]
struct MpRectF {
    left: c_float,
    top: c_float,
    bottom: c_float,
    right: c_float,
}

#[repr(C)]
struct MpImageProcessingOptions {
    has_region_of_interest: c_int,
    region_of_interest: MpRectF,
    rotation_degrees: c_int,
}

#[repr(C)]
struct MpCategory {
    index: c_int,
    score: c_float,
    category_name: *const c_char,
    display_name: *const c_char,
}

#[repr(C)]
struct MpCategories {
    categories: *const MpCategory,
    categories_count: u32,
}

#[repr(C)]
struct MpNormalizedKeypoint {
    x: c_float,
    y: c_float,
    label: *const c_char,
    score: c_float,
    has_score: bool,
}

#[repr(C)]
struct MpDetection {
    categories: *const MpCategory,
    categories_count: u32,
    bounding_box: MpRect,
    keypoints: *const MpNormalizedKeypoint,
    keypoints_count: u32,
}

#[repr(C)]
struct MpDetectionResult {
    detections: *const MpDetection,
    detections_count: u32,
}

#[repr(C)]
struct MpNormalizedLandmark {
    x: c_float,
    y: c_float,
    z: c_float,
    has_visibility: bool,
    visibility: c_float,
    has_presence: bool,
    presence: c_float,
    name: *const c_char,
}

#[repr(C)]
struct MpNormalizedLandmarks {
    landmarks: *const MpNormalizedLandmark,
    landmarks_count: u32,
}

#[repr(C)]
struct MpLandmark {
    x: c_float,
    y: c_float,
    z: c_float,
    has_visibility: bool,
    visibility: c_float,
    has_presence: bool,
    presence: c_float,
    name: *const c_char,
}

#[repr(C)]
struct MpLandmarks {
    landmarks: *const MpLandmark,
    landmarks_count: u32,
}

#[repr(C)]
struct MpPoseLandmarkerOptions {
    base_options: MpBaseOptions,
    running_mode: c_uint,
    num_poses: c_int,
    min_pose_detection_confidence: c_float,
    min_pose_presence_confidence: c_float,
    min_tracking_confidence: c_float,
    output_segmentation_masks: bool,
    result_callback: *const c_void,
}

#[repr(C)]
struct MpPoseLandmarkerResult {
    segmentation_masks: *const *mut c_void,
    segmentation_masks_count: u32,
    pose_landmarks: *const MpNormalizedLandmarks,
    pose_landmarks_count: u32,
    pose_world_landmarks: *const MpLandmarks,
    pose_world_landmarks_count: u32,
}

#[repr(C)]
struct MpObjectDetectorOptions {
    base_options: MpBaseOptions,
    running_mode: c_uint,
    display_names_locale: *const c_char,
    max_results: c_int,
    score_threshold: c_float,
    category_allowlist: *const *const c_char,
    category_allowlist_count: u32,
    category_denylist: *const *const c_char,
    category_denylist_count: u32,
    result_callback: *const c_void,
}

#[repr(C)]
struct MpGestureRecognizerOptions {
    base_options: MpBaseOptions,
    running_mode: c_uint,
    num_hands: c_int,
    min_hand_detection_confidence: c_float,
    min_hand_presence_confidence: c_float,
    min_tracking_confidence: c_float,
    canned_gestures_classifier_options: MpClassifierOptions,
    custom_gestures_classifier_options: MpClassifierOptions,
    result_callback: *const c_void,
}

#[repr(C)]
struct MpGestureRecognizerResult {
    gestures: *const MpCategories,
    gestures_count: u32,
    handedness: *const MpCategories,
    handedness_count: u32,
    hand_landmarks: *const MpNormalizedLandmarks,
    hand_landmarks_count: u32,
    hand_world_landmarks: *const MpLandmarks,
    hand_world_landmarks_count: u32,
}

#[repr(C)]
struct MpAudioClassifierOptions {
    base_options: MpBaseOptions,
    classifier_options: MpClassifierOptions,
    running_mode: c_uint,
    result_callback: *const c_void,
}

#[repr(C)]
struct MpClassifications {
    categories: *const MpCategory,
    categories_count: u32,
    head_index: c_int,
    head_name: *const c_char,
}

#[repr(C)]
struct MpClassificationResult {
    classifications: *const MpClassifications,
    classifications_count: u32,
    timestamp_ms: i64,
    has_timestamp_ms: bool,
}

#[repr(C)]
struct MpAudioClassifierResult {
    results: *const MpClassificationResult,
    results_count: c_int,
}

#[repr(C)]
struct MpAudioData {
    num_channels: c_int,
    sample_rate: c_double,
    audio_data: *const c_float,
    audio_data_size: usize,
}

type FnErrorFree = unsafe extern "C" fn(*mut c_char);
type FnImageCreateUint8 = unsafe extern "C" fn(
    c_uint,
    c_int,
    c_int,
    *const u8,
    c_int,
    *mut *mut c_void,
    *mut *mut c_char,
) -> c_uint;
type FnImageFree = unsafe extern "C" fn(*mut c_void);

type FnPoseCreate = unsafe extern "C" fn(
    *const MpPoseLandmarkerOptions,
    *mut *mut c_void,
    *mut *mut c_char,
) -> c_uint;
type FnPoseDetectImage = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *const MpImageProcessingOptions,
    *mut MpPoseLandmarkerResult,
    *mut *mut c_char,
) -> c_uint;
type FnPoseDetectVideo = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *const MpImageProcessingOptions,
    i64,
    *mut MpPoseLandmarkerResult,
    *mut *mut c_char,
) -> c_uint;
type FnPoseCloseResult = unsafe extern "C" fn(*mut MpPoseLandmarkerResult);
type FnPoseClose = unsafe extern "C" fn(*mut c_void, *mut *mut c_char) -> c_uint;

type FnObjectCreate = unsafe extern "C" fn(
    *const MpObjectDetectorOptions,
    *mut *mut c_void,
    *mut *mut c_char,
) -> c_uint;
type FnObjectDetectImage = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *const MpImageProcessingOptions,
    *mut MpDetectionResult,
    *mut *mut c_char,
) -> c_uint;
type FnObjectDetectVideo = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *const MpImageProcessingOptions,
    i64,
    *mut MpDetectionResult,
    *mut *mut c_char,
) -> c_uint;
type FnObjectCloseResult = unsafe extern "C" fn(*mut MpDetectionResult);
type FnObjectClose = unsafe extern "C" fn(*mut c_void, *mut *mut c_char) -> c_uint;

type FnGestureCreate = unsafe extern "C" fn(
    *const MpGestureRecognizerOptions,
    *mut *mut c_void,
    *mut *mut c_char,
) -> c_uint;
type FnGestureImage = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *const MpImageProcessingOptions,
    *mut MpGestureRecognizerResult,
    *mut *mut c_char,
) -> c_uint;
type FnGestureVideo = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *const MpImageProcessingOptions,
    i64,
    *mut MpGestureRecognizerResult,
    *mut *mut c_char,
) -> c_uint;
type FnGestureCloseResult = unsafe extern "C" fn(*mut MpGestureRecognizerResult);
type FnGestureClose = unsafe extern "C" fn(*mut c_void, *mut *mut c_char) -> c_uint;

type FnAudioCreate = unsafe extern "C" fn(
    *const MpAudioClassifierOptions,
    *mut *mut c_void,
    *mut *mut c_char,
) -> c_uint;
type FnAudioClassify = unsafe extern "C" fn(
    *mut c_void,
    *const MpAudioData,
    *mut MpAudioClassifierResult,
    *mut *mut c_char,
) -> c_uint;
type FnAudioCloseResult = unsafe extern "C" fn(*mut MpAudioClassifierResult);
type FnAudioClose = unsafe extern "C" fn(*mut c_void, *mut *mut c_char) -> c_uint;

struct Symbols {
    _lib: Library,
    error_free: FnErrorFree,
    image_create_uint8: FnImageCreateUint8,
    image_free: FnImageFree,
    pose_create: FnPoseCreate,
    pose_detect_image: FnPoseDetectImage,
    pose_detect_video: FnPoseDetectVideo,
    pose_close_result: FnPoseCloseResult,
    pose_close: FnPoseClose,
    object_create: FnObjectCreate,
    object_detect_image: FnObjectDetectImage,
    object_detect_video: FnObjectDetectVideo,
    object_close_result: FnObjectCloseResult,
    object_close: FnObjectClose,
    gesture_create: FnGestureCreate,
    gesture_image: FnGestureImage,
    gesture_video: FnGestureVideo,
    gesture_close_result: FnGestureCloseResult,
    gesture_close: FnGestureClose,
    audio_create: FnAudioCreate,
    audio_classify: FnAudioClassify,
    audio_close_result: FnAudioCloseResult,
    audio_close: FnAudioClose,
}

/// Loaded native `libmediapipe` shared library handle.
#[derive(Clone)]
pub struct NativeRuntime {
    symbols: Arc<Symbols>,
    library_path: PathBuf,
}

impl fmt::Debug for NativeRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeRuntime")
            .field("library_path", &self.library_path)
            .finish()
    }
}

impl NativeRuntime {
    /// Load `libmediapipe` from `MEDIAPIPE_LIBRARY_PATH` or standard OS search locations.
    ///
    /// # Errors
    ///
    /// Returns [`NativeError::RuntimeUnavailable`] when the library cannot be opened.
    pub fn load_default() -> Result<Self, NativeError> {
        if let Ok(explicit) = std::env::var("MEDIAPIPE_LIBRARY_PATH")
            && !explicit.trim().is_empty()
        {
            return Self::load_from(Path::new(&explicit));
        }
        let candidates = default_library_names();
        let mut last_err = String::from("no candidate library names for target");
        for name in candidates {
            match Self::load_from(Path::new(name)) {
                Ok(rt) => return Ok(rt),
                Err(err) => last_err = err.to_string(),
            }
        }
        Err(NativeError::RuntimeUnavailable(last_err))
    }

    /// Load `libmediapipe` from an explicit filesystem path or library soname.
    ///
    /// # Errors
    ///
    /// Returns [`NativeError::RuntimeUnavailable`] or [`NativeError::MissingSymbol`].
    pub fn load_from(path: &Path) -> Result<Self, NativeError> {
        // SAFETY: Loading a platform shared library and resolving MediaPipe C ABI symbols.
        let lib = unsafe { Library::new(path) }
            .map_err(|err| NativeError::RuntimeUnavailable(format!("{}: {err}", path.display())))?;
        let symbols = load_symbols(lib)?;
        Ok(Self {
            symbols: Arc::new(symbols),
            library_path: path.to_path_buf(),
        })
    }

    /// Path or library name that was loaded.
    #[must_use]
    pub fn library_path(&self) -> &Path {
        &self.library_path
    }

    /// Instantiate a [`NativePoseLandmarker`] from verified model bytes or path.
    ///
    /// # Errors
    ///
    /// Returns [`NativeError`] when MediaPipe refuses the model or options.
    pub fn create_pose_landmarker(
        &self,
        model_path: &Path,
        options: &PoseLandmarkerOptions,
    ) -> Result<NativePoseLandmarker, NativeError> {
        let path_c = path_to_cstring(model_path)?;
        let c_opts = MpPoseLandmarkerOptions {
            base_options: MpBaseOptions {
                model_asset_path: path_c.as_ptr(),
                ..MpBaseOptions::default()
            },
            running_mode: options.running_mode.code(),
            num_poses: options.num_poses,
            min_pose_detection_confidence: options.min_pose_detection_confidence,
            min_pose_presence_confidence: options.min_pose_presence_confidence,
            min_tracking_confidence: options.min_tracking_confidence,
            output_segmentation_masks: options.output_segmentation_masks,
            result_callback: ptr::null(),
        };
        let mut handle: *mut c_void = ptr::null_mut();
        let mut err_msg: *mut c_char = ptr::null_mut();
        // SAFETY: All pointers remain valid across the synchronous creation call.
        let status = unsafe { (self.symbols.pose_create)(&c_opts, &mut handle, &mut err_msg) };
        check_status(&self.symbols, status, err_msg)?;
        Ok(NativePoseLandmarker {
            runtime: self.clone(),
            handle,
            mode: options.running_mode,
        })
    }

    /// Instantiate a [`NativeObjectDetector`] from a verified model path.
    ///
    /// # Errors
    ///
    /// Returns [`NativeError`] when MediaPipe refuses the model or options.
    pub fn create_object_detector(
        &self,
        model_path: &Path,
        options: &ObjectDetectorOptions,
    ) -> Result<NativeObjectDetector, NativeError> {
        let path_c = path_to_cstring(model_path)?;
        let lists = CClassifierLists::new(&options.classifier)?;
        let c_opts = MpObjectDetectorOptions {
            base_options: MpBaseOptions {
                model_asset_path: path_c.as_ptr(),
                ..MpBaseOptions::default()
            },
            running_mode: options.running_mode.code(),
            display_names_locale: ptr::null(),
            max_results: options.classifier.max_results,
            score_threshold: options.classifier.score_threshold,
            category_allowlist: lists.allow_ptr(),
            category_allowlist_count: lists.allow_len(),
            category_denylist: lists.deny_ptr(),
            category_denylist_count: lists.deny_len(),
            result_callback: ptr::null(),
        };
        let mut handle: *mut c_void = ptr::null_mut();
        let mut err_msg: *mut c_char = ptr::null_mut();
        // SAFETY: All pointers remain valid across the synchronous creation call.
        let status = unsafe { (self.symbols.object_create)(&c_opts, &mut handle, &mut err_msg) };
        check_status(&self.symbols, status, err_msg)?;
        Ok(NativeObjectDetector {
            runtime: self.clone(),
            handle,
            mode: options.running_mode,
        })
    }

    /// Instantiate a [`NativeGestureRecognizer`] from a verified model path.
    ///
    /// # Errors
    ///
    /// Returns [`NativeError`] when MediaPipe refuses the model or options.
    pub fn create_gesture_recognizer(
        &self,
        model_path: &Path,
        options: &GestureRecognizerOptions,
    ) -> Result<NativeGestureRecognizer, NativeError> {
        let path_c = path_to_cstring(model_path)?;
        let lists = CClassifierLists::new(&options.canned_gestures)?;
        let c_opts = MpGestureRecognizerOptions {
            base_options: MpBaseOptions {
                model_asset_path: path_c.as_ptr(),
                ..MpBaseOptions::default()
            },
            running_mode: options.running_mode.code(),
            num_hands: options.num_hands,
            min_hand_detection_confidence: options.min_hand_detection_confidence,
            min_hand_presence_confidence: options.min_hand_presence_confidence,
            min_tracking_confidence: options.min_tracking_confidence,
            canned_gestures_classifier_options: lists.as_c_options(&options.canned_gestures),
            custom_gestures_classifier_options: MpClassifierOptions::default(),
            result_callback: ptr::null(),
        };
        let mut handle: *mut c_void = ptr::null_mut();
        let mut err_msg: *mut c_char = ptr::null_mut();
        // SAFETY: All pointers remain valid across the synchronous creation call.
        let status = unsafe { (self.symbols.gesture_create)(&c_opts, &mut handle, &mut err_msg) };
        check_status(&self.symbols, status, err_msg)?;
        Ok(NativeGestureRecognizer {
            runtime: self.clone(),
            handle,
            mode: options.running_mode,
        })
    }

    /// Instantiate a [`NativeAudioClassifier`] in audio-clips mode.
    ///
    /// # Errors
    ///
    /// Returns [`NativeError`] when MediaPipe refuses the model or options.
    pub fn create_audio_classifier(
        &self,
        model_path: &Path,
        options: &AudioClassifierOptions,
    ) -> Result<NativeAudioClassifier, NativeError> {
        let path_c = path_to_cstring(model_path)?;
        let lists = CClassifierLists::new(&options.classifier)?;
        let c_opts = MpAudioClassifierOptions {
            base_options: MpBaseOptions {
                model_asset_path: path_c.as_ptr(),
                ..MpBaseOptions::default()
            },
            classifier_options: lists.as_c_options(&options.classifier),
            running_mode: 1, // kMpAudioRunningModeAudioClips
            result_callback: ptr::null(),
        };
        let mut handle: *mut c_void = ptr::null_mut();
        let mut err_msg: *mut c_char = ptr::null_mut();
        // SAFETY: All pointers remain valid across the synchronous creation call.
        let status = unsafe { (self.symbols.audio_create)(&c_opts, &mut handle, &mut err_msg) };
        check_status(&self.symbols, status, err_msg)?;
        Ok(NativeAudioClassifier {
            runtime: self.clone(),
            handle,
        })
    }

    fn make_image(&self, image: &MpImage) -> Result<OwnedMpImage<'_>, NativeError> {
        let mut out: *mut c_void = ptr::null_mut();
        let mut err_msg: *mut c_char = ptr::null_mut();
        let data_len = c_int::try_from(image.data().len())
            .map_err(|_| NativeError::InvalidParameter("image buffer exceeds c_int".to_owned()))?;
        // SAFETY: `image.data()` is contiguous and has length `data_len`.
        let status = unsafe {
            (self.symbols.image_create_uint8)(
                image.format().c_format(),
                image.width() as c_int,
                image.height() as c_int,
                image.data().as_ptr(),
                data_len,
                &mut out,
                &mut err_msg,
            )
        };
        check_status(&self.symbols, status, err_msg)?;
        Ok(OwnedMpImage {
            symbols: &self.symbols,
            ptr: out,
        })
    }
}

struct OwnedMpImage<'a> {
    symbols: &'a Symbols,
    ptr: *mut c_void,
}

impl Drop for OwnedMpImage<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` was created by `MpImageCreateFromUint8Data`.
            unsafe { (self.symbols.image_free)(self.ptr) };
        }
    }
}

/// Native pose landmarker backed by `libmediapipe`.
pub struct NativePoseLandmarker {
    runtime: NativeRuntime,
    handle: *mut c_void,
    mode: RunningMode,
}

// SAFETY: MediaPipe task handles may be moved across threads when synchronized by caller.
unsafe impl Send for NativePoseLandmarker {}

impl NativePoseLandmarker {
    /// Detect pose landmarks on a single image (`RunningMode::Image`).
    ///
    /// # Errors
    ///
    /// Returns [`NativeError`] on wrong running mode or native inference failure.
    pub fn detect_image(
        &mut self,
        image: &MpImage,
        rotation_degrees: i32,
    ) -> Result<PoseLandmarkerResult, NativeError> {
        if self.mode != RunningMode::Image {
            return Err(NativeError::InvalidParameter(
                "detect_image requires RunningMode::Image".to_owned(),
            ));
        }
        let mp_img = self.runtime.make_image(image)?;
        let proc_opts = image_processing_options(rotation_degrees);
        let mut raw = zeroed_pose_result();
        let mut err_msg: *mut c_char = ptr::null_mut();
        // SAFETY: Handle and pointers are valid for the duration of the call.
        let status = unsafe {
            (self.runtime.symbols.pose_detect_image)(
                self.handle,
                mp_img.ptr,
                &proc_opts,
                &mut raw,
                &mut err_msg,
            )
        };
        check_status(&self.runtime.symbols, status, err_msg)?;
        let converted = unsafe { convert_pose_result(&raw) };
        unsafe { (self.runtime.symbols.pose_close_result)(&mut raw) };
        Ok(converted)
    }

    /// Detect pose landmarks on a video frame (`RunningMode::Video`).
    ///
    /// # Errors
    ///
    /// Returns [`NativeError`] on wrong running mode or native inference failure.
    pub fn detect_for_video(
        &mut self,
        image: &MpImage,
        rotation_degrees: i32,
        timestamp_ms: i64,
    ) -> Result<PoseLandmarkerResult, NativeError> {
        if self.mode != RunningMode::Video {
            return Err(NativeError::InvalidParameter(
                "detect_for_video requires RunningMode::Video".to_owned(),
            ));
        }
        let mp_img = self.runtime.make_image(image)?;
        let proc_opts = image_processing_options(rotation_degrees);
        let mut raw = zeroed_pose_result();
        let mut err_msg: *mut c_char = ptr::null_mut();
        // SAFETY: Handle and pointers are valid for the duration of the call.
        let status = unsafe {
            (self.runtime.symbols.pose_detect_video)(
                self.handle,
                mp_img.ptr,
                &proc_opts,
                timestamp_ms,
                &mut raw,
                &mut err_msg,
            )
        };
        check_status(&self.runtime.symbols, status, err_msg)?;
        let converted = unsafe { convert_pose_result(&raw) };
        unsafe { (self.runtime.symbols.pose_close_result)(&mut raw) };
        Ok(converted)
    }
}

impl Drop for NativePoseLandmarker {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let mut err_msg: *mut c_char = ptr::null_mut();
            // SAFETY: Releases the native handle once.
            let _ = unsafe { (self.runtime.symbols.pose_close)(self.handle, &mut err_msg) };
            free_error(&self.runtime.symbols, err_msg);
        }
    }
}

/// Native object detector backed by `libmediapipe`.
pub struct NativeObjectDetector {
    runtime: NativeRuntime,
    handle: *mut c_void,
    mode: RunningMode,
}

// SAFETY: MediaPipe task handles may be moved across threads when synchronized by caller.
unsafe impl Send for NativeObjectDetector {}

impl NativeObjectDetector {
    /// Detect objects on a single image (`RunningMode::Image`).
    ///
    /// # Errors
    ///
    /// Returns [`NativeError`] on wrong running mode or native inference failure.
    pub fn detect_image(
        &mut self,
        image: &MpImage,
        rotation_degrees: i32,
    ) -> Result<ObjectDetectorResult, NativeError> {
        if self.mode != RunningMode::Image {
            return Err(NativeError::InvalidParameter(
                "detect_image requires RunningMode::Image".to_owned(),
            ));
        }
        let mp_img = self.runtime.make_image(image)?;
        let proc_opts = image_processing_options(rotation_degrees);
        let mut raw = MpDetectionResult {
            detections: ptr::null(),
            detections_count: 0,
        };
        let mut err_msg: *mut c_char = ptr::null_mut();
        let status = unsafe {
            (self.runtime.symbols.object_detect_image)(
                self.handle,
                mp_img.ptr,
                &proc_opts,
                &mut raw,
                &mut err_msg,
            )
        };
        check_status(&self.runtime.symbols, status, err_msg)?;
        let converted = unsafe { convert_detection_result(&raw) };
        unsafe { (self.runtime.symbols.object_close_result)(&mut raw) };
        Ok(converted)
    }

    /// Detect objects on a video frame (`RunningMode::Video`).
    ///
    /// # Errors
    ///
    /// Returns [`NativeError`] on wrong running mode or native inference failure.
    pub fn detect_for_video(
        &mut self,
        image: &MpImage,
        rotation_degrees: i32,
        timestamp_ms: i64,
    ) -> Result<ObjectDetectorResult, NativeError> {
        if self.mode != RunningMode::Video {
            return Err(NativeError::InvalidParameter(
                "detect_for_video requires RunningMode::Video".to_owned(),
            ));
        }
        let mp_img = self.runtime.make_image(image)?;
        let proc_opts = image_processing_options(rotation_degrees);
        let mut raw = MpDetectionResult {
            detections: ptr::null(),
            detections_count: 0,
        };
        let mut err_msg: *mut c_char = ptr::null_mut();
        let status = unsafe {
            (self.runtime.symbols.object_detect_video)(
                self.handle,
                mp_img.ptr,
                &proc_opts,
                timestamp_ms,
                &mut raw,
                &mut err_msg,
            )
        };
        check_status(&self.runtime.symbols, status, err_msg)?;
        let converted = unsafe { convert_detection_result(&raw) };
        unsafe { (self.runtime.symbols.object_close_result)(&mut raw) };
        Ok(converted)
    }
}

impl Drop for NativeObjectDetector {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let mut err_msg: *mut c_char = ptr::null_mut();
            let _ = unsafe { (self.runtime.symbols.object_close)(self.handle, &mut err_msg) };
            free_error(&self.runtime.symbols, err_msg);
        }
    }
}

/// Native gesture recognizer backed by `libmediapipe`.
pub struct NativeGestureRecognizer {
    runtime: NativeRuntime,
    handle: *mut c_void,
    mode: RunningMode,
}

// SAFETY: MediaPipe task handles may be moved across threads when synchronized by caller.
unsafe impl Send for NativeGestureRecognizer {}

impl NativeGestureRecognizer {
    /// Recognize hand gestures on a single image (`RunningMode::Image`).
    ///
    /// # Errors
    ///
    /// Returns [`NativeError`] on wrong running mode or native inference failure.
    pub fn recognize_image(
        &mut self,
        image: &MpImage,
        rotation_degrees: i32,
    ) -> Result<GestureRecognizerResult, NativeError> {
        if self.mode != RunningMode::Image {
            return Err(NativeError::InvalidParameter(
                "recognize_image requires RunningMode::Image".to_owned(),
            ));
        }
        let mp_img = self.runtime.make_image(image)?;
        let proc_opts = image_processing_options(rotation_degrees);
        let mut raw = zeroed_gesture_result();
        let mut err_msg: *mut c_char = ptr::null_mut();
        let status = unsafe {
            (self.runtime.symbols.gesture_image)(
                self.handle,
                mp_img.ptr,
                &proc_opts,
                &mut raw,
                &mut err_msg,
            )
        };
        check_status(&self.runtime.symbols, status, err_msg)?;
        let converted = unsafe { convert_gesture_result(&raw) };
        unsafe { (self.runtime.symbols.gesture_close_result)(&mut raw) };
        Ok(converted)
    }

    /// Recognize hand gestures on a video frame (`RunningMode::Video`).
    ///
    /// # Errors
    ///
    /// Returns [`NativeError`] on wrong running mode or native inference failure.
    pub fn recognize_for_video(
        &mut self,
        image: &MpImage,
        rotation_degrees: i32,
        timestamp_ms: i64,
    ) -> Result<GestureRecognizerResult, NativeError> {
        if self.mode != RunningMode::Video {
            return Err(NativeError::InvalidParameter(
                "recognize_for_video requires RunningMode::Video".to_owned(),
            ));
        }
        let mp_img = self.runtime.make_image(image)?;
        let proc_opts = image_processing_options(rotation_degrees);
        let mut raw = zeroed_gesture_result();
        let mut err_msg: *mut c_char = ptr::null_mut();
        let status = unsafe {
            (self.runtime.symbols.gesture_video)(
                self.handle,
                mp_img.ptr,
                &proc_opts,
                timestamp_ms,
                &mut raw,
                &mut err_msg,
            )
        };
        check_status(&self.runtime.symbols, status, err_msg)?;
        let converted = unsafe { convert_gesture_result(&raw) };
        unsafe { (self.runtime.symbols.gesture_close_result)(&mut raw) };
        Ok(converted)
    }
}

impl Drop for NativeGestureRecognizer {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let mut err_msg: *mut c_char = ptr::null_mut();
            let _ = unsafe { (self.runtime.symbols.gesture_close)(self.handle, &mut err_msg) };
            free_error(&self.runtime.symbols, err_msg);
        }
    }
}

/// Native audio classifier backed by `libmediapipe`.
pub struct NativeAudioClassifier {
    runtime: NativeRuntime,
    handle: *mut c_void,
}

// SAFETY: MediaPipe task handles may be moved across threads when synchronized by caller.
unsafe impl Send for NativeAudioClassifier {}

impl NativeAudioClassifier {
    /// Classify a PCM audio clip.
    ///
    /// # Errors
    ///
    /// Returns [`NativeError`] when `audio` is invalid or native classification fails.
    pub fn classify(&mut self, audio: &AudioBuffer) -> Result<AudioClassifierResult, NativeError> {
        if audio.channels == 0 || audio.sample_rate_hz <= 0.0 || audio.samples.is_empty() {
            return Err(NativeError::InvalidParameter(
                "audio buffer must have positive channels, sample rate, and samples".to_owned(),
            ));
        }
        let c_audio = MpAudioData {
            num_channels: audio.channels as c_int,
            sample_rate: audio.sample_rate_hz,
            audio_data: audio.samples.as_ptr(),
            audio_data_size: audio.samples.len(),
        };
        let mut raw = MpAudioClassifierResult {
            results: ptr::null(),
            results_count: 0,
        };
        let mut err_msg: *mut c_char = ptr::null_mut();
        let status = unsafe {
            (self.runtime.symbols.audio_classify)(self.handle, &c_audio, &mut raw, &mut err_msg)
        };
        check_status(&self.runtime.symbols, status, err_msg)?;
        let converted = unsafe { convert_audio_result(&raw) };
        unsafe { (self.runtime.symbols.audio_close_result)(&mut raw) };
        Ok(converted)
    }
}

impl Drop for NativeAudioClassifier {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let mut err_msg: *mut c_char = ptr::null_mut();
            let _ = unsafe { (self.runtime.symbols.audio_close)(self.handle, &mut err_msg) };
            free_error(&self.runtime.symbols, err_msg);
        }
    }
}

// --- FFI helpers -----------------------------------------------------------

fn default_library_names() -> &'static [&'static str] {
    if cfg!(target_os = "macos") {
        &["libmediapipe.dylib", "/usr/local/lib/libmediapipe.dylib"]
    } else if cfg!(target_os = "windows") {
        &["mediapipe.dll"]
    } else {
        &["libmediapipe.so", "/usr/local/lib/libmediapipe.so"]
    }
}

fn load_symbols(lib: Library) -> Result<Symbols, NativeError> {
    macro_rules! sym {
        ($name:literal, $ty:ty) => {{
            let symbol = unsafe { lib.get::<$ty>($name) }.map_err(|_| {
                NativeError::MissingSymbol(String::from_utf8_lossy($name).into_owned())
            })?;
            *symbol
        }};
    }
    let error_free = sym!(b"MpErrorFree\0", FnErrorFree);
    let image_create_uint8 = sym!(b"MpImageCreateFromUint8Data\0", FnImageCreateUint8);
    let image_free = sym!(b"MpImageFree\0", FnImageFree);
    let pose_create = sym!(b"MpPoseLandmarkerCreate\0", FnPoseCreate);
    let pose_detect_image = sym!(b"MpPoseLandmarkerDetectImage\0", FnPoseDetectImage);
    let pose_detect_video = sym!(b"MpPoseLandmarkerDetectForVideo\0", FnPoseDetectVideo);
    let pose_close_result = sym!(b"MpPoseLandmarkerCloseResult\0", FnPoseCloseResult);
    let pose_close = sym!(b"MpPoseLandmarkerClose\0", FnPoseClose);
    let object_create = sym!(b"MpObjectDetectorCreate\0", FnObjectCreate);
    let object_detect_image = sym!(b"MpObjectDetectorDetectImage\0", FnObjectDetectImage);
    let object_detect_video = sym!(b"MpObjectDetectorDetectForVideo\0", FnObjectDetectVideo);
    let object_close_result = sym!(b"MpObjectDetectorCloseResult\0", FnObjectCloseResult);
    let object_close = sym!(b"MpObjectDetectorClose\0", FnObjectClose);
    let gesture_create = sym!(b"MpGestureRecognizerCreate\0", FnGestureCreate);
    let gesture_image = sym!(b"MpGestureRecognizerRecognizeImage\0", FnGestureImage);
    let gesture_video = sym!(b"MpGestureRecognizerRecognizeForVideo\0", FnGestureVideo);
    let gesture_close_result = sym!(b"MpGestureRecognizerCloseResult\0", FnGestureCloseResult);
    let gesture_close = sym!(b"MpGestureRecognizerClose\0", FnGestureClose);
    let audio_create = sym!(b"MpAudioClassifierCreate\0", FnAudioCreate);
    let audio_classify = sym!(b"MpAudioClassifierClassify\0", FnAudioClassify);
    let audio_close_result = sym!(b"MpAudioClassifierCloseResult\0", FnAudioCloseResult);
    let audio_close = sym!(b"MpAudioClassifierClose\0", FnAudioClose);
    Ok(Symbols {
        _lib: lib,
        error_free,
        image_create_uint8,
        image_free,
        pose_create,
        pose_detect_image,
        pose_detect_video,
        pose_close_result,
        pose_close,
        object_create,
        object_detect_image,
        object_detect_video,
        object_close_result,
        object_close,
        gesture_create,
        gesture_image,
        gesture_video,
        gesture_close_result,
        gesture_close,
        audio_create,
        audio_classify,
        audio_close_result,
        audio_close,
    })
}

struct CClassifierLists {
    _allow_owned: Vec<CString>,
    allow_ptrs: Vec<*const c_char>,
    _deny_owned: Vec<CString>,
    deny_ptrs: Vec<*const c_char>,
}

impl CClassifierLists {
    fn new(opts: &ClassifierOptions) -> Result<Self, NativeError> {
        let mut allow_owned = Vec::with_capacity(opts.category_allowlist.len());
        let mut allow_ptrs = Vec::with_capacity(opts.category_allowlist.len());
        for label in &opts.category_allowlist {
            let cs = CString::new(label.as_str()).map_err(|_| {
                NativeError::InvalidParameter("interior NUL in allowlist".to_owned())
            })?;
            allow_ptrs.push(cs.as_ptr());
            allow_owned.push(cs);
        }
        let mut deny_owned = Vec::with_capacity(opts.category_denylist.len());
        let mut deny_ptrs = Vec::with_capacity(opts.category_denylist.len());
        for label in &opts.category_denylist {
            let cs = CString::new(label.as_str()).map_err(|_| {
                NativeError::InvalidParameter("interior NUL in denylist".to_owned())
            })?;
            deny_ptrs.push(cs.as_ptr());
            deny_owned.push(cs);
        }
        Ok(Self {
            _allow_owned: allow_owned,
            allow_ptrs,
            _deny_owned: deny_owned,
            deny_ptrs,
        })
    }

    fn allow_ptr(&self) -> *const *const c_char {
        if self.allow_ptrs.is_empty() {
            ptr::null()
        } else {
            self.allow_ptrs.as_ptr()
        }
    }

    fn allow_len(&self) -> u32 {
        self.allow_ptrs.len() as u32
    }

    fn deny_ptr(&self) -> *const *const c_char {
        if self.deny_ptrs.is_empty() {
            ptr::null()
        } else {
            self.deny_ptrs.as_ptr()
        }
    }

    fn deny_len(&self) -> u32 {
        self.deny_ptrs.len() as u32
    }

    fn as_c_options(&self, opts: &ClassifierOptions) -> MpClassifierOptions {
        MpClassifierOptions {
            display_names_locale: ptr::null(),
            max_results: opts.max_results,
            score_threshold: opts.score_threshold,
            category_allowlist: self.allow_ptr(),
            category_allowlist_count: self.allow_len(),
            category_denylist: self.deny_ptr(),
            category_denylist_count: self.deny_len(),
        }
    }
}

fn path_to_cstring(path: &Path) -> Result<CString, NativeError> {
    let text = path
        .to_str()
        .ok_or_else(|| NativeError::InvalidParameter("model path is not valid UTF-8".to_owned()))?;
    CString::new(text)
        .map_err(|_| NativeError::InvalidParameter("model path contains interior NUL".to_owned()))
}

fn image_processing_options(rotation_degrees: i32) -> MpImageProcessingOptions {
    MpImageProcessingOptions {
        has_region_of_interest: 0,
        region_of_interest: MpRectF {
            left: 0.0,
            top: 0.0,
            bottom: 0.0,
            right: 0.0,
        },
        rotation_degrees,
    }
}

fn check_status(
    symbols: &Symbols,
    status: c_uint,
    err_msg: *mut c_char,
) -> Result<(), NativeError> {
    if status == 0 {
        free_error(symbols, err_msg);
        return Ok(());
    }
    let message = if err_msg.is_null() {
        String::from("mediapipe call failed")
    } else {
        // SAFETY: `err_msg` was allocated by MediaPipe when non-null.
        let text = unsafe { CStr::from_ptr(err_msg) }
            .to_string_lossy()
            .into_owned();
        free_error(symbols, err_msg);
        text
    };
    Err(NativeError::Status {
        code: status,
        message,
    })
}

fn free_error(symbols: &Symbols, err_msg: *mut c_char) {
    if !err_msg.is_null() {
        // SAFETY: `err_msg` came from MediaPipe C API and is freed once.
        unsafe { (symbols.error_free)(err_msg) };
    }
}

fn zeroed_pose_result() -> MpPoseLandmarkerResult {
    MpPoseLandmarkerResult {
        segmentation_masks: ptr::null(),
        segmentation_masks_count: 0,
        pose_landmarks: ptr::null(),
        pose_landmarks_count: 0,
        pose_world_landmarks: ptr::null(),
        pose_world_landmarks_count: 0,
    }
}

fn zeroed_gesture_result() -> MpGestureRecognizerResult {
    MpGestureRecognizerResult {
        gestures: ptr::null(),
        gestures_count: 0,
        handedness: ptr::null(),
        handedness_count: 0,
        hand_landmarks: ptr::null(),
        hand_landmarks_count: 0,
        hand_world_landmarks: ptr::null(),
        hand_world_landmarks_count: 0,
    }
}

unsafe fn optional_cstr(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

unsafe fn convert_categories(ptr: *const MpCategory, count: u32) -> Vec<Category> {
    if ptr.is_null() || count == 0 {
        return Vec::new();
    }
    let slice = unsafe { std::slice::from_raw_parts(ptr, count as usize) };
    slice
        .iter()
        .map(|cat| Category {
            index: cat.index,
            score: cat.score,
            category_name: unsafe { optional_cstr(cat.category_name) },
            display_name: unsafe { optional_cstr(cat.display_name) },
        })
        .collect()
}

unsafe fn convert_normalized_landmarks_list(
    ptr: *const MpNormalizedLandmarks,
    count: u32,
) -> Vec<Vec<NormalizedLandmark>> {
    if ptr.is_null() || count == 0 {
        return Vec::new();
    }
    let outer = unsafe { std::slice::from_raw_parts(ptr, count as usize) };
    outer
        .iter()
        .map(|list| {
            if list.landmarks.is_null() || list.landmarks_count == 0 {
                return Vec::new();
            }
            let items = unsafe {
                std::slice::from_raw_parts(list.landmarks, list.landmarks_count as usize)
            };
            items
                .iter()
                .map(|lm| {
                    NormalizedLandmark::new(f64::from(lm.x), f64::from(lm.y), f64::from(lm.z))
                        .with_confidence(
                            lm.has_visibility.then_some(lm.visibility),
                            lm.has_presence.then_some(lm.presence),
                        )
                })
                .collect()
        })
        .collect()
}

unsafe fn convert_world_landmarks_list(
    ptr: *const MpLandmarks,
    count: u32,
) -> Vec<Vec<WorldLandmark>> {
    if ptr.is_null() || count == 0 {
        return Vec::new();
    }
    let outer = unsafe { std::slice::from_raw_parts(ptr, count as usize) };
    outer
        .iter()
        .map(|list| {
            if list.landmarks.is_null() || list.landmarks_count == 0 {
                return Vec::new();
            }
            let items = unsafe {
                std::slice::from_raw_parts(list.landmarks, list.landmarks_count as usize)
            };
            items
                .iter()
                .map(|lm| WorldLandmark {
                    x: lm.x,
                    y: lm.y,
                    z: lm.z,
                    visibility: lm.has_visibility.then_some(lm.visibility),
                    presence: lm.has_presence.then_some(lm.presence),
                })
                .collect()
        })
        .collect()
}

unsafe fn convert_pose_result(raw: &MpPoseLandmarkerResult) -> PoseLandmarkerResult {
    PoseLandmarkerResult {
        pose_landmarks: unsafe {
            convert_normalized_landmarks_list(raw.pose_landmarks, raw.pose_landmarks_count)
        },
        pose_world_landmarks: unsafe {
            convert_world_landmarks_list(raw.pose_world_landmarks, raw.pose_world_landmarks_count)
        },
    }
}

unsafe fn convert_detection_result(raw: &MpDetectionResult) -> ObjectDetectorResult {
    if raw.detections.is_null() || raw.detections_count == 0 {
        return ObjectDetectorResult::default();
    }
    let slice =
        unsafe { std::slice::from_raw_parts(raw.detections, raw.detections_count as usize) };
    let detections = slice
        .iter()
        .map(|det| {
            let categories = unsafe { convert_categories(det.categories, det.categories_count) };
            let keypoints = if det.keypoints.is_null() || det.keypoints_count == 0 {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(det.keypoints, det.keypoints_count as usize) }
                    .iter()
                    .map(|kp| crate::tasks::NormalizedKeypoint {
                        x: kp.x,
                        y: kp.y,
                        score: kp.has_score.then_some(kp.score),
                    })
                    .collect()
            };
            let width = det.bounding_box.right.saturating_sub(det.bounding_box.left);
            let height = det.bounding_box.bottom.saturating_sub(det.bounding_box.top);
            Detection {
                categories,
                bounding_box: BoundingBox::new(
                    f64::from(det.bounding_box.left),
                    f64::from(det.bounding_box.top),
                    f64::from(width),
                    f64::from(height),
                ),
                keypoints,
            }
        })
        .collect();
    ObjectDetectorResult { detections }
}

unsafe fn convert_gesture_result(raw: &MpGestureRecognizerResult) -> GestureRecognizerResult {
    let convert_cat_groups = |ptr: *const MpCategories, count: u32| -> Vec<Vec<Category>> {
        if ptr.is_null() || count == 0 {
            return Vec::new();
        }
        unsafe { std::slice::from_raw_parts(ptr, count as usize) }
            .iter()
            .map(|group| unsafe { convert_categories(group.categories, group.categories_count) })
            .collect()
    };
    GestureRecognizerResult {
        gestures: convert_cat_groups(raw.gestures, raw.gestures_count),
        handedness: convert_cat_groups(raw.handedness, raw.handedness_count),
        hand_landmarks: unsafe {
            convert_normalized_landmarks_list(raw.hand_landmarks, raw.hand_landmarks_count)
        },
        hand_world_landmarks: unsafe {
            convert_world_landmarks_list(raw.hand_world_landmarks, raw.hand_world_landmarks_count)
        },
    }
}

unsafe fn convert_audio_result(raw: &MpAudioClassifierResult) -> AudioClassifierResult {
    if raw.results.is_null() || raw.results_count <= 0 {
        return AudioClassifierResult::default();
    }
    let slice = unsafe { std::slice::from_raw_parts(raw.results, raw.results_count as usize) };
    let results = slice
        .iter()
        .map(|entry| {
            let heads = if entry.classifications.is_null() || entry.classifications_count == 0 {
                Vec::new()
            } else {
                unsafe {
                    std::slice::from_raw_parts(
                        entry.classifications,
                        entry.classifications_count as usize,
                    )
                }
                .iter()
                .map(|head| ClassificationHead {
                    head_index: head.head_index,
                    head_name: unsafe { optional_cstr(head.head_name) },
                    categories: unsafe {
                        convert_categories(head.categories, head.categories_count)
                    },
                })
                .collect()
            };
            Classifications {
                timestamp_ms: entry.has_timestamp_ms.then_some(entry.timestamp_ms),
                heads,
            }
        })
        .collect();
    AudioClassifierResult { results }
}

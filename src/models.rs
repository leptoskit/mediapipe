//! SHA-256 and byte-length verified MediaPipe model pins and local cache.
//!
//! Model files are treated as untrusted bytes until both their exact length and
//! SHA-256 digest match the pinned values. Verification runs on write and again
//! on every load from disk so tampered cache files are refused.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Allowed immutable HTTPS prefix for official MediaPipe model downloads.
pub const ALLOWED_MODEL_PREFIX: &str = "https://storage.googleapis.com/mediapipe-models/";

/// Maximum pinned model size accepted by [`ModelPin::validate`] (64 MiB).
pub const MAX_MODEL_BYTES: usize = 64 * 1024 * 1024;

/// Immutable pinned MediaPipe model metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelPin {
    /// Stable model identifier (for example `mediapipe/pose_landmarker_lite/float16/1`).
    pub id: String,
    /// Immutable HTTPS URL under [`ALLOWED_MODEL_PREFIX`].
    pub url: String,
    /// Lowercase 64-character hex SHA-256 digest.
    pub sha256: String,
    /// Exact byte length of the pinned file.
    pub byte_length: usize,
}

/// Standard BlazePose GHUM Lite model (33 body landmarks).
#[must_use]
pub fn pose_landmarker_lite_pin() -> ModelPin {
    ModelPin {
        id: "mediapipe/pose_landmarker_lite/float16/1".to_owned(),
        url: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task".to_owned(),
        sha256: "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a".to_owned(),
        byte_length: 5_777_746,
    }
}

/// Standard EfficientDet-Lite0 int8 COCO object detector.
#[must_use]
pub fn efficientdet_lite0_pin() -> ModelPin {
    ModelPin {
        id: "mediapipe/efficientdet_lite0/int8/1".to_owned(),
        url: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite".to_owned(),
        sha256: "0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb".to_owned(),
        byte_length: 4_602_795,
    }
}

/// Standard MediaPipe canned hand gesture recognizer.
#[must_use]
pub fn gesture_recognizer_pin() -> ModelPin {
    ModelPin {
        id: "mediapipe/gesture_recognizer/float16/1".to_owned(),
        url: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task".to_owned(),
        sha256: "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482".to_owned(),
        byte_length: 8_373_440,
    }
}

/// Standard YAMNet AudioSet classifier.
#[must_use]
pub fn yamnet_pin() -> ModelPin {
    ModelPin {
        id: "mediapipe/yamnet/float32/1".to_owned(),
        url: "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite".to_owned(),
        sha256: "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf".to_owned(),
        byte_length: 4_126_810,
    }
}

/// Why a model pin or cached model file was rejected.
#[derive(Debug)]
pub enum ModelError {
    /// Pin metadata failed URL, digest, or length validation.
    InvalidPin(&'static str),
    /// Byte length did not match the pin.
    LengthMismatch { expected: usize, actual: usize },
    /// SHA-256 digest did not match the pin.
    DigestMismatch { expected: String, actual: String },
    /// Cached model was not found.
    NotCached(PathBuf),
    /// Filesystem I/O failure.
    Io(io::Error),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPin(reason) => write!(f, "invalid model pin: {reason}"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "model byte length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::DigestMismatch { expected, actual } => {
                write!(
                    f,
                    "model sha256 mismatch: expected {expected}, got {actual}"
                )
            }
            Self::NotCached(path) => write!(f, "pinned model not cached at {}", path.display()),
            Self::Io(err) => write!(f, "model cache I/O error: {err}"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<io::Error> for ModelError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl ModelPin {
    /// Validate that the pin uses the official HTTPS bucket, a 64-hex SHA-256,
    /// and a positive bounded length.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidPin`] when any pin invariant fails.
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.id.trim().is_empty() {
            return Err(ModelError::InvalidPin("model id must not be empty"));
        }
        if !self.url.starts_with(ALLOWED_MODEL_PREFIX)
            || self.url.contains("..")
            || self.url.contains('?')
            || self.url.contains('#')
            || self.url.contains('@')
        {
            return Err(ModelError::InvalidPin(
                "model url must be an immutable mediapipe-models HTTPS path",
            ));
        }
        if self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        {
            return Err(ModelError::InvalidPin(
                "model sha256 must be 64 lowercase hex characters",
            ));
        }
        if self.byte_length == 0 || self.byte_length > MAX_MODEL_BYTES {
            return Err(ModelError::InvalidPin("model byte_length out of bounds"));
        }
        Ok(())
    }

    /// Verify in-memory bytes against both `byte_length` and `sha256`.
    ///
    /// # Errors
    ///
    /// Fails on invalid pin, wrong byte length, or digest mismatch.
    pub fn verify_bytes(&self, bytes: &[u8]) -> Result<(), ModelError> {
        self.validate()?;
        if bytes.len() != self.byte_length {
            return Err(ModelError::LengthMismatch {
                expected: self.byte_length,
                actual: bytes.len(),
            });
        }
        let actual = sha256_hex(bytes);
        if actual != self.sha256 {
            return Err(ModelError::DigestMismatch {
                expected: self.sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    /// Read and verify a model file from disk.
    ///
    /// # Errors
    ///
    /// Fails if the file cannot be read or does not match `byte_length` and `sha256`.
    pub fn verify_file(&self, path: &Path) -> Result<Vec<u8>, ModelError> {
        let bytes = fs::read(path)?;
        self.verify_bytes(&bytes)?;
        Ok(bytes)
    }
}

/// Directory-backed local cache of SHA-256-verified MediaPipe model files.
#[derive(Debug, Clone)]
pub struct ModelCache {
    root: PathBuf,
}

impl ModelCache {
    /// Open or target a model cache directory.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Cache directory path.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Deterministic file path for a validated pin (`<root>/<sha256>.model`).
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidPin`] when the pin fails validation.
    pub fn path_for(&self, pin: &ModelPin) -> Result<PathBuf, ModelError> {
        pin.validate()?;
        Ok(self.root.join(format!("{}.model", pin.sha256)))
    }

    /// Verify `bytes` against `pin` and atomically persist them into the cache.
    ///
    /// # Errors
    ///
    /// Rejects unverified bytes before writing anything to disk.
    pub fn store_verified(&self, pin: &ModelPin, bytes: &[u8]) -> Result<PathBuf, ModelError> {
        pin.verify_bytes(bytes)?;
        fs::create_dir_all(&self.root)?;
        let final_path = self.path_for(pin)?;
        let tmp_path = self.root.join(format!("{}.tmp", pin.sha256));
        fs::write(&tmp_path, bytes)?;
        fs::rename(&tmp_path, &final_path)?;
        Ok(final_path)
    }

    /// Load a cached model and re-verify its exact length and SHA-256 digest.
    ///
    /// If the cached file was modified or truncated, it is removed and an error is returned.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::NotCached`] when absent, or mismatch errors on tampering.
    pub fn load_verified(&self, pin: &ModelPin) -> Result<(PathBuf, Vec<u8>), ModelError> {
        let path = self.path_for(pin)?;
        if !path.exists() {
            return Err(ModelError::NotCached(path));
        }
        match pin.verify_file(&path) {
            Ok(bytes) => Ok((path, bytes)),
            Err(err @ (ModelError::LengthMismatch { .. } | ModelError::DigestMismatch { .. })) => {
                let _ = fs::remove_file(&path);
                Err(err)
            }
            Err(err) => Err(err),
        }
    }
}

/// Compute lowercase hex SHA-256 of `bytes`.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

//! LeptosKit bridge protocol (`"mediapipe"`), Wasm guest client, and native Rust host plugin.
//!
//! Inference runs strictly on the native host; raw camera pixels never cross
//! Wasm linear memory. Guests send a compact [`SessionConfig`] naming the
//! camera session and SHA-256-pinned model files, then poll compact
//! [`VisionObservation`] records (<1 KiB/frame) containing upright landmarks,
//! object detections, and canned gestures.

use leptoskit_bridge::codec::{Decoder, Encoder};
use leptoskit_bridge::{Bytes, Error, MethodId};

use crate::overlay::{BoundingBox, NormalizedLandmark};
use crate::tasks::{
    Category, Detection, GestureRecognizerResult, ObjectDetectorResult, PoseLandmarkerResult,
    VisionObservation,
};

/// Runtime registration name in `leptoskit.toml` and `[package.metadata.leptoskit.runtime]`.
pub const NAME: &str = "mediapipe";
/// `() -> Status` (availability flags and backend capabilities).
pub const STATUS: MethodId = MethodId(0);
/// `SessionConfig -> SessionId` (starts host-side vision tracking).
pub const START_SESSION: MethodId = MethodId(1);
/// `SessionId -> ()` (stops tracking and frees host tasks).
pub const STOP_SESSION: MethodId = MethodId(2);
/// `SessionId -> Option<VisionObservation>` (returns latest observation if available).
pub const POLL_OBSERVATION: MethodId = MethodId(3);
/// Supported wire methods.
pub const METHODS: &[MethodId] = &[STATUS, START_SESSION, STOP_SESSION, POLL_OBSERVATION];

const WIRE_VERSION: u32 = 1;
const MAX_STRING_BYTES: usize = 2048;
const MAX_ALLOWLIST_ITEMS: usize = 128;

/// Host-assigned vision tracking session identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VisionSessionId(pub u32);

impl VisionSessionId {
    /// Encode as 4 little-endian bytes.
    #[must_use]
    pub fn encode(self) -> Bytes {
        let mut enc = Encoder::with_capacity(4);
        enc.u32(self.0);
        enc.finish()
    }

    /// Decode a session identifier.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] on malformed or trailing bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        let mut dec = Decoder::new(bytes);
        let id = dec.u32()?;
        dec.finish()?;
        Ok(Self(id))
    }
}

/// Host MediaPipe runtime capability summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeStatus {
    /// Wire protocol version (`1`).
    pub version: u32,
    /// Whether the host can run pose landmarking.
    pub pose_available: bool,
    /// Whether the host can run object detection.
    pub object_available: bool,
    /// Whether the host can run hand gesture recognition.
    pub gesture_available: bool,
}

impl RuntimeStatus {
    /// Encode as wire bytes.
    #[must_use]
    pub fn encode(self) -> Bytes {
        let mut enc = Encoder::with_capacity(8);
        enc.u32(self.version);
        let mut flags = 0u32;
        if self.pose_available {
            flags |= 1 << 0;
        }
        if self.object_available {
            flags |= 1 << 1;
        }
        if self.gesture_available {
            flags |= 1 << 2;
        }
        enc.u32(flags);
        enc.finish()
    }

    /// Decode from wire bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] on unknown version or trailing bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        let mut dec = Decoder::new(bytes);
        let version = dec.u32()?;
        if version != WIRE_VERSION {
            return Err(Error::InvalidArgument);
        }
        let flags = dec.u32()?;
        dec.finish()?;
        Ok(Self {
            version,
            pose_available: (flags & (1 << 0)) != 0,
            object_available: (flags & (1 << 1)) != 0,
            gesture_available: (flags & (1 << 2)) != 0,
        })
    }
}

/// Configuration for a host-managed live MediaPipe vision session.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionConfig {
    /// Host camera session id to attach frame analysis to (`0` for standalone).
    pub camera_session_id: u32,
    /// Enable BlazePose 33-landmark tracking.
    pub enable_pose: bool,
    /// Enable object detection (for example `"sports ball"` or local-check labels).
    pub enable_objects: bool,
    /// Enable hand gesture recognition.
    pub enable_gestures: bool,
    /// Optional local verified path or pin id for the pose model.
    pub pose_model: String,
    /// Optional local verified path or pin id for the object detector model.
    pub object_model: String,
    /// Optional local verified path or pin id for the gesture recognizer model.
    pub gesture_model: String,
    /// Score threshold for detections and gestures.
    pub score_threshold: f32,
    /// Maximum detections per frame.
    pub max_results: u32,
    /// Category allowlist for object/gesture filtering.
    pub category_allowlist: Vec<String>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            camera_session_id: 0,
            enable_pose: true,
            enable_objects: false,
            enable_gestures: false,
            pose_model: String::new(),
            object_model: String::new(),
            gesture_model: String::new(),
            score_threshold: 0.2,
            max_results: 3,
            category_allowlist: Vec::new(),
        }
    }
}

impl SessionConfig {
    /// Encode into wire bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if strings or allowlist exceed limits.
    pub fn encode(&self) -> Result<Bytes, Error> {
        if self.pose_model.len() > MAX_STRING_BYTES
            || self.object_model.len() > MAX_STRING_BYTES
            || self.gesture_model.len() > MAX_STRING_BYTES
            || self.category_allowlist.len() > MAX_ALLOWLIST_ITEMS
        {
            return Err(Error::InvalidArgument);
        }
        let mut enc = Encoder::with_capacity(64);
        enc.u32(WIRE_VERSION);
        enc.u32(self.camera_session_id);
        let mut flags = 0u32;
        if self.enable_pose {
            flags |= 1 << 0;
        }
        if self.enable_objects {
            flags |= 1 << 1;
        }
        if self.enable_gestures {
            flags |= 1 << 2;
        }
        enc.u32(flags);
        enc.bytes(self.pose_model.as_bytes())?;
        enc.bytes(self.object_model.as_bytes())?;
        enc.bytes(self.gesture_model.as_bytes())?;
        enc.u32(self.score_threshold.to_bits());
        enc.u32(self.max_results);
        enc.u32(self.category_allowlist.len() as u32);
        for item in &self.category_allowlist {
            if item.len() > MAX_STRING_BYTES {
                return Err(Error::InvalidArgument);
            }
            enc.bytes(item.as_bytes())?;
        }
        Ok(enc.finish())
    }

    /// Decode from wire bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] on malformed input.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        let mut dec = Decoder::new(bytes);
        let version = dec.u32()?;
        if version != WIRE_VERSION {
            return Err(Error::InvalidArgument);
        }
        let camera_session_id = dec.u32()?;
        let flags = dec.u32()?;
        let pose_model = dec.string()?.to_owned();
        let object_model = dec.string()?.to_owned();
        let gesture_model = dec.string()?.to_owned();
        let score_threshold = f32::from_bits(dec.u32()?);
        if !score_threshold.is_finite() {
            return Err(Error::InvalidArgument);
        }
        let max_results = dec.u32()?;
        let allow_len = dec.u32()? as usize;
        if allow_len > MAX_ALLOWLIST_ITEMS {
            return Err(Error::InvalidArgument);
        }
        let mut category_allowlist = Vec::with_capacity(allow_len);
        for _ in 0..allow_len {
            category_allowlist.push(dec.string()?.to_owned());
        }
        dec.finish()?;
        Ok(Self {
            camera_session_id,
            enable_pose: (flags & (1 << 0)) != 0,
            enable_objects: (flags & (1 << 1)) != 0,
            enable_gestures: (flags & (1 << 2)) != 0,
            pose_model,
            object_model,
            gesture_model,
            score_threshold,
            max_results,
            category_allowlist,
        })
    }
}

/// Encode an optional [`VisionObservation`] into compact wire bytes.
///
/// # Errors
///
/// Returns [`Error::TooLarge`] if payloads overflow wire limits.
pub fn encode_observation(obs: Option<&VisionObservation>) -> Result<Bytes, Error> {
    let mut enc = Encoder::with_capacity(128);
    let Some(obs) = obs else {
        enc.u32(0);
        return Ok(enc.finish());
    };
    enc.u32(1);
    let ts_bits = obs.timestamp_ms.to_le_bytes();
    enc.u32(u32::from_le_bytes([
        ts_bits[0], ts_bits[1], ts_bits[2], ts_bits[3],
    ]));
    enc.u32(u32::from_le_bytes([
        ts_bits[4], ts_bits[5], ts_bits[6], ts_bits[7],
    ]));
    enc.u32(obs.frame_width);
    enc.u32(obs.frame_height);
    enc.u32(obs.rotation_degrees as u32);
    enc.u32(u32::from(obs.mirrored));

    // Pose landmarks (first detected pose for compact live stream)
    match obs.pose.as_ref().and_then(|p| p.pose_landmarks.first()) {
        Some(landmarks) => {
            enc.u32(landmarks.len() as u32);
            for lm in landmarks {
                enc.u32((lm.x as f32).to_bits());
                enc.u32((lm.y as f32).to_bits());
                enc.u32((lm.z as f32).to_bits());
                enc.u32(lm.visibility.unwrap_or(-1.0).to_bits());
                enc.u32(lm.presence.unwrap_or(-1.0).to_bits());
            }
        }
        None => enc.u32(0),
    }

    // Object detections
    match obs.objects.as_ref() {
        Some(objs) => {
            enc.u32(objs.detections.len() as u32);
            for det in &objs.detections {
                enc.u32((det.bounding_box.left as f32).to_bits());
                enc.u32((det.bounding_box.top as f32).to_bits());
                enc.u32((det.bounding_box.width as f32).to_bits());
                enc.u32((det.bounding_box.height as f32).to_bits());
                enc.u32(det.categories.len() as u32);
                for cat in &det.categories {
                    enc.u32(cat.index as u32);
                    enc.u32(cat.score.to_bits());
                    enc.bytes(cat.category_name.as_deref().unwrap_or("").as_bytes())?;
                }
            }
        }
        None => enc.u32(0),
    }

    // Hand gestures
    match obs.gestures.as_ref() {
        Some(gestures) => {
            enc.u32(gestures.gestures.len() as u32);
            for hand in &gestures.gestures {
                enc.u32(hand.len() as u32);
                for cat in hand {
                    enc.u32(cat.index as u32);
                    enc.u32(cat.score.to_bits());
                    enc.bytes(cat.category_name.as_deref().unwrap_or("").as_bytes())?;
                }
            }
            enc.u32(gestures.hand_landmarks.len() as u32);
        }
        None => {
            enc.u32(0);
            enc.u32(0);
        }
    }

    Ok(enc.finish())
}

/// Decode an optional [`VisionObservation`] from wire bytes.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] on malformed input.
pub fn decode_observation(bytes: &[u8]) -> Result<Option<VisionObservation>, Error> {
    let mut dec = Decoder::new(bytes);
    let present = dec.u32()?;
    if present == 0 {
        dec.finish()?;
        return Ok(None);
    }
    if present != 1 {
        return Err(Error::InvalidArgument);
    }
    let ts_lo = dec.u32()?.to_le_bytes();
    let ts_hi = dec.u32()?.to_le_bytes();
    let timestamp_ms = i64::from_le_bytes([
        ts_lo[0], ts_lo[1], ts_lo[2], ts_lo[3], ts_hi[0], ts_hi[1], ts_hi[2], ts_hi[3],
    ]);
    let frame_width = dec.u32()?;
    let frame_height = dec.u32()?;
    let rotation_degrees = dec.u32()? as i32;
    let mirrored = match dec.u32()? {
        0 => false,
        1 => true,
        _ => return Err(Error::InvalidArgument),
    };

    let pose_count = dec.u32()? as usize;
    if pose_count > 128 {
        return Err(Error::InvalidArgument);
    }
    let pose = if pose_count == 0 {
        None
    } else {
        let mut landmarks = Vec::with_capacity(pose_count);
        for _ in 0..pose_count {
            let x = f64::from(f32::from_bits(dec.u32()?));
            let y = f64::from(f32::from_bits(dec.u32()?));
            let z = f64::from(f32::from_bits(dec.u32()?));
            let vis = f32::from_bits(dec.u32()?);
            let pres = f32::from_bits(dec.u32()?);
            landmarks.push(
                NormalizedLandmark::new(x, y, z)
                    .with_confidence((vis >= 0.0).then_some(vis), (pres >= 0.0).then_some(pres)),
            );
        }
        Some(PoseLandmarkerResult {
            pose_landmarks: vec![landmarks],
            pose_world_landmarks: Vec::new(),
        })
    };

    let det_count = dec.u32()? as usize;
    if det_count > 64 {
        return Err(Error::InvalidArgument);
    }
    let objects = if det_count == 0 {
        None
    } else {
        let mut detections = Vec::with_capacity(det_count);
        for _ in 0..det_count {
            let left = f64::from(f32::from_bits(dec.u32()?));
            let top = f64::from(f32::from_bits(dec.u32()?));
            let width = f64::from(f32::from_bits(dec.u32()?));
            let height = f64::from(f32::from_bits(dec.u32()?));
            let cat_count = dec.u32()? as usize;
            if cat_count > 32 {
                return Err(Error::InvalidArgument);
            }
            let mut categories = Vec::with_capacity(cat_count);
            for _ in 0..cat_count {
                let index = dec.u32()? as i32;
                let score = f32::from_bits(dec.u32()?);
                let name = dec.string()?;
                categories.push(Category {
                    index,
                    score,
                    category_name: (!name.is_empty()).then(|| name.to_owned()),
                    display_name: None,
                });
            }
            detections.push(Detection {
                categories,
                bounding_box: BoundingBox::new(left, top, width, height),
                keypoints: Vec::new(),
            });
        }
        Some(ObjectDetectorResult { detections })
    };

    let hand_group_count = dec.u32()? as usize;
    if hand_group_count > 8 {
        return Err(Error::InvalidArgument);
    }
    let mut gesture_groups = Vec::with_capacity(hand_group_count);
    for _ in 0..hand_group_count {
        let cat_count = dec.u32()? as usize;
        if cat_count > 16 {
            return Err(Error::InvalidArgument);
        }
        let mut hand = Vec::with_capacity(cat_count);
        for _ in 0..cat_count {
            let index = dec.u32()? as i32;
            let score = f32::from_bits(dec.u32()?);
            let name = dec.string()?;
            hand.push(Category {
                index,
                score,
                category_name: (!name.is_empty()).then(|| name.to_owned()),
                display_name: None,
            });
        }
        gesture_groups.push(hand);
    }
    let hand_landmarks_count = dec.u32()? as usize;
    if hand_landmarks_count > 8 {
        return Err(Error::InvalidArgument);
    }
    dec.finish()?;

    let gestures = if gesture_groups.is_empty() && hand_landmarks_count == 0 {
        None
    } else {
        Some(GestureRecognizerResult {
            gestures: gesture_groups,
            handedness: Vec::new(),
            hand_landmarks: vec![Vec::new(); hand_landmarks_count],
            hand_world_landmarks: Vec::new(),
        })
    };

    Ok(Some(VisionObservation {
        timestamp_ms,
        frame_width,
        frame_height,
        rotation_degrees,
        mirrored,
        pose,
        objects,
        gestures,
    }))
}

/// Wasm guest client for calling the host `"mediapipe"` plugin.
#[cfg(all(feature = "leptoskit-guest", target_arch = "wasm32"))]
pub mod guest {
    use leptoskit_bridge::Error;
    use leptoskit_bridge::codec::Decoder;
    use leptoskit_bridge::guest::Plugin;

    use super::{
        POLL_OBSERVATION, RuntimeStatus, START_SESSION, STATUS, STOP_SESSION, SessionConfig,
        VisionSessionId, decode_observation,
    };
    use crate::tasks::VisionObservation;

    /// Handle to the installed host MediaPipe plugin.
    #[derive(Clone, Copy)]
    pub struct GuestMediaPipe(Plugin);

    impl GuestMediaPipe {
        /// Resolve the `"mediapipe"` plugin on the active bridge.
        ///
        /// # Errors
        ///
        /// Returns [`Error::Unavailable`] when the host did not register MediaPipe.
        pub fn new() -> Result<Self, Error> {
            Plugin::resolve(super::NAME).map(Self)
        }

        /// Query host MediaPipe capabilities.
        ///
        /// # Errors
        ///
        /// Returns bridge errors if the host plugin is unavailable or malformed.
        pub async fn status(self) -> Result<RuntimeStatus, Error> {
            let reply = self.0.call(STATUS, &[])?.await?;
            RuntimeStatus::decode(&reply)
        }

        /// Start a host-side vision tracking session.
        ///
        /// # Errors
        ///
        /// Returns bridge errors if the model is missing or the runtime is unavailable.
        pub async fn start_session(self, config: &SessionConfig) -> Result<VisionSessionId, Error> {
            let payload = config.encode()?;
            let reply = self.0.call(START_SESSION, &payload)?.await?;
            VisionSessionId::decode(&reply)
        }

        /// Poll the latest compact frame observation from an active session.
        ///
        /// # Errors
        ///
        /// Returns bridge errors on unknown session or wire failure.
        pub async fn poll_observation(
            self,
            session: VisionSessionId,
        ) -> Result<Option<VisionObservation>, Error> {
            let reply = self.0.call(POLL_OBSERVATION, &session.encode())?.await?;
            decode_observation(&reply)
        }

        /// Stop and release an active vision tracking session.
        ///
        /// # Errors
        ///
        /// Returns bridge errors on unknown session.
        pub async fn stop_session(self, session: VisionSessionId) -> Result<(), Error> {
            let reply = self.0.call(STOP_SESSION, &session.encode())?.await?;
            Decoder::new(&reply).finish()
        }
    }
}

/// Native Rust host plugin for Linux/macOS/Windows LeptosKit hosts.
#[cfg(feature = "leptoskit-host")]
mod host_impl {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use leptoskit_bridge::{Bytes, Error, MethodId, Plugin};

    use super::{
        METHODS, NAME, POLL_OBSERVATION, RuntimeStatus, START_SESSION, STATUS, STOP_SESSION,
        SessionConfig, VisionSessionId, WIRE_VERSION, encode_observation,
    };
    use crate::native::NativeRuntime;
    use crate::tasks::VisionObservation;

    struct ActiveSession {
        _config: SessionConfig,
        latest: Option<VisionObservation>,
    }

    /// Host-side `"mediapipe"` plugin implementation for Rust desktop hosts.
    pub struct HostMediaPipe {
        _data_dir: PathBuf,
        _cache_dir: PathBuf,
        runtime: Option<NativeRuntime>,
        next_id: u32,
        sessions: BTreeMap<u32, ActiveSession>,
    }

    impl HostMediaPipe {
        /// Create a host plugin probing `libmediapipe` (honestly reporting unavailable if absent).
        #[must_use]
        pub fn new(data_dir: &Path, cache_dir: &Path) -> Self {
            Self {
                _data_dir: data_dir.to_path_buf(),
                _cache_dir: cache_dir.to_path_buf(),
                runtime: NativeRuntime::load_default().ok(),
                next_id: 1,
                sessions: BTreeMap::new(),
            }
        }

        /// Inject an observation for testing or desktop camera adapters.
        pub fn push_observation(&mut self, session: VisionSessionId, obs: VisionObservation) {
            if let Some(entry) = self.sessions.get_mut(&session.0) {
                entry.latest = Some(obs);
            }
        }
    }

    impl Plugin for HostMediaPipe {
        fn name(&self) -> &'static str {
            NAME
        }

        fn methods(&self) -> &'static [MethodId] {
            METHODS
        }

        fn invoke(&mut self, method: MethodId, payload: Bytes) -> Result<Bytes, Error> {
            match method {
                STATUS => {
                    if !payload.is_empty() {
                        return Err(Error::InvalidArgument);
                    }
                    let ready = self.runtime.is_some();
                    Ok(RuntimeStatus {
                        version: WIRE_VERSION,
                        pose_available: ready,
                        object_available: ready,
                        gesture_available: ready,
                    }
                    .encode())
                }
                START_SESSION => {
                    let config = SessionConfig::decode(&payload)?;
                    if self.runtime.is_none() {
                        return Err(Error::Unavailable);
                    }
                    let id = self.next_id;
                    self.next_id = self.next_id.checked_add(1).ok_or(Error::Exhausted)?;
                    self.sessions.insert(
                        id,
                        ActiveSession {
                            _config: config,
                            latest: None,
                        },
                    );
                    Ok(VisionSessionId(id).encode())
                }
                STOP_SESSION => {
                    let session = VisionSessionId::decode(&payload)?;
                    self.sessions
                        .remove(&session.0)
                        .ok_or(Error::InvalidArgument)?;
                    Ok(Bytes::new())
                }
                POLL_OBSERVATION => {
                    let session = VisionSessionId::decode(&payload)?;
                    let entry = self
                        .sessions
                        .get_mut(&session.0)
                        .ok_or(Error::InvalidArgument)?;
                    let obs = entry.latest.take();
                    encode_observation(obs.as_ref())
                }
                _ => Err(Error::Unavailable),
            }
        }
    }

    /// Factory referenced by `[package.metadata.leptoskit.runtime.*.rust-factory]`.
    #[must_use]
    pub fn host_plugin(data_dir: &Path, cache_dir: &Path) -> HostMediaPipe {
        HostMediaPipe::new(data_dir, cache_dir)
    }
}

#[cfg(feature = "leptoskit-host")]
pub use host_impl::{HostMediaPipe, host_plugin};

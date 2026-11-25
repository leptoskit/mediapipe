use mediapipe::{
    FramePacer, ImageFormat, MpImage, NormalizedLandmark, PoseJoint, RunningMode, TimestampTracker,
};

#[test]
fn normalized_landmark_preserves_optional_confidence() {
    let bare = NormalizedLandmark::new(0.25, 0.75, -0.1);
    assert_eq!(bare.visibility, None);
    assert_eq!(bare.presence, None);

    let confident = bare.with_confidence(Some(0.92), Some(0.88));
    assert_eq!(confident.visibility, Some(0.92));
    assert_eq!(confident.presence, Some(0.88));
}

#[test]
fn image_conversions_and_quarter_turn_rotation() {
    // 2x1 BGRA -> SRGBA
    let bgra = [10, 20, 30, 255, 40, 50, 60, 128];
    let rgba = MpImage::from_bgra8(2, 1, &bgra).expect("bgra8");
    assert_eq!(rgba.format(), ImageFormat::Srgba);
    assert_eq!(rgba.data(), &[30, 20, 10, 255, 60, 50, 40, 128]);

    // 90 degrees clockwise swaps 2x1 to 1x2
    let rotated = rgba.rotate_clockwise(90).expect("rotate 90");
    assert_eq!((rotated.width(), rotated.height()), (1, 2));
    assert_eq!(rotated.data(), &[30, 20, 10, 255, 60, 50, 40, 128]);

    // 2x2 NV21 converts without error
    let nv21 = [16, 235, 128, 128, 128, 128];
    let srgb = MpImage::from_nv21(2, 2, &nv21).expect("nv21");
    assert_eq!(srgb.format(), ImageFormat::Srgb);
    assert_eq!(srgb.data().len(), 2 * 2 * 3);
}

#[test]
fn timestamp_tracker_and_frame_pacer_enforce_monotonic_bounds() {
    let mut tracker = TimestampTracker::new();
    assert_eq!(tracker.next(10), 10);
    assert_eq!(tracker.next(10), 11);
    assert_eq!(tracker.next(5), 12);
    assert_eq!(tracker.next(40), 40);

    let mut pacer = FramePacer::new(30);
    assert!(pacer.try_acquire(100));
    assert!(!pacer.try_acquire(110)); // busy
    pacer.release();
    assert!(!pacer.try_acquire(120)); // interval < 30ms
    assert!(pacer.try_acquire(130));
    assert_eq!(pacer.dropped_frames(), 2);
}

#[test]
fn pose_joint_indices_cover_blazepose_33() {
    assert_eq!(PoseJoint::COUNT, 33);
    assert_eq!(PoseJoint::Nose.index(), 0);
    assert_eq!(PoseJoint::LeftHip.index(), 23);
    assert_eq!(PoseJoint::RightFootIndex.index(), 32);
    assert_eq!(
        RunningMode::from_code(RunningMode::Video.code()),
        Some(RunningMode::Video)
    );
}

#[cfg(feature = "model-cache")]
#[test]
fn model_cache_verifies_pin_and_rejects_tampering() {
    use mediapipe::models::{
        ModelCache, ModelError, ModelPin, efficientdet_lite0_pin, gesture_recognizer_pin,
        pose_landmarker_lite_pin, sha256_hex, yamnet_pin,
    };

    for pin in [
        pose_landmarker_lite_pin(),
        efficientdet_lite0_pin(),
        gesture_recognizer_pin(),
        yamnet_pin(),
    ] {
        pin.validate().expect("standard pin valid");
    }

    let bytes = b"test-mediapipe-model-bytes";
    let pin = ModelPin {
        id: "mediapipe/test/1".to_owned(),
        url: "https://storage.googleapis.com/mediapipe-models/test/model.task".to_owned(),
        sha256: sha256_hex(bytes),
        byte_length: bytes.len(),
    };

    let dir = std::env::temp_dir().join(format!("mp_cache_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cache = ModelCache::new(&dir);
    let path = cache.store_verified(&pin, bytes).expect("store");
    let (_, loaded) = cache.load_verified(&pin).expect("load");
    assert_eq!(loaded, bytes);

    // Tamper with the cached file -> rejected and purged
    std::fs::write(&path, b"corrupted-mediapipe-bytes!").expect("corrupt");
    assert!(matches!(
        cache.load_verified(&pin),
        Err(ModelError::DigestMismatch { .. })
    ));
    assert!(!path.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(feature = "native")]
#[test]
fn native_runtime_fails_closed_when_library_absent() {
    use std::path::Path;

    use mediapipe::native::{NATIVE_ARTIFACTS, NativeError, NativeRuntime};

    assert_eq!(NATIVE_ARTIFACTS.len(), 4);
    let err = NativeRuntime::load_from(Path::new("/nonexistent/libmediapipe.so"))
        .expect_err("missing library fails closed");
    assert!(matches!(err, NativeError::RuntimeUnavailable(_)));
}

#[cfg(feature = "leptoskit-host")]
#[test]
fn bridge_codec_and_host_plugin_round_trip() {
    use std::path::Path;

    use leptoskit_bridge::{Bytes, Error, Plugin};
    use mediapipe::bridge::{
        RuntimeStatus, START_SESSION, STATUS, SessionConfig, decode_observation,
        encode_observation, host_plugin,
    };
    use mediapipe::{
        BoundingBox, Category, Detection, GestureRecognizerResult, ObjectDetectorResult,
        PoseLandmarkerResult, VisionObservation,
    };

    let cfg = SessionConfig {
        camera_session_id: 7,
        enable_pose: true,
        enable_objects: true,
        enable_gestures: false,
        pose_model: "/cache/pose.task".to_owned(),
        object_model: "/cache/det.tflite".to_owned(),
        gesture_model: String::new(),
        score_threshold: 0.25,
        max_results: 3,
        category_allowlist: vec!["sports ball".to_owned()],
    };
    let encoded_cfg = cfg.encode().expect("encode cfg");
    assert_eq!(
        SessionConfig::decode(&encoded_cfg).expect("decode cfg"),
        cfg
    );

    let obs = VisionObservation {
        timestamp_ms: 1_234_567,
        frame_width: 480,
        frame_height: 640,
        rotation_degrees: 90,
        mirrored: true,
        pose: Some(PoseLandmarkerResult {
            pose_landmarks: vec![vec![
                NormalizedLandmark::new(0.5, 0.25, -0.05).with_confidence(Some(0.95), Some(0.9)),
            ]],
            pose_world_landmarks: Vec::new(),
        }),
        objects: Some(ObjectDetectorResult {
            detections: vec![Detection {
                categories: vec![Category {
                    index: 36,
                    score: 0.85,
                    category_name: Some("sports ball".to_owned()),
                    display_name: None,
                }],
                bounding_box: BoundingBox::new(100.0, 200.0, 48.0, 48.0),
                keypoints: Vec::new(),
            }],
        }),
        gestures: Some(GestureRecognizerResult {
            gestures: vec![vec![Category {
                index: 1,
                score: 0.91,
                category_name: Some("Thumb_Up".to_owned()),
                display_name: None,
            }]],
            handedness: Vec::new(),
            hand_landmarks: vec![Vec::new()],
            hand_world_landmarks: Vec::new(),
        }),
    };
    let wire = encode_observation(Some(&obs)).expect("encode obs");
    let decoded = decode_observation(&wire)
        .expect("decode obs")
        .expect("present");
    assert_eq!(decoded.timestamp_ms, obs.timestamp_ms);
    assert_eq!(decoded.frame_width, 480);
    assert_eq!(decoded.frame_height, 640);
    assert_eq!(decoded.rotation_degrees, 90);
    assert!(decoded.mirrored);
    assert_eq!(
        decoded.pose.unwrap().pose_landmarks[0][0].visibility,
        Some(0.95)
    );
    assert_eq!(
        decoded.objects.unwrap().detections[0].categories[0]
            .category_name
            .as_deref(),
        Some("sports ball")
    );
    assert_eq!(
        decoded.gestures.unwrap().gestures[0][0]
            .category_name
            .as_deref(),
        Some("Thumb_Up")
    );

    let mut plugin = host_plugin(Path::new("/tmp"), Path::new("/tmp"));
    let status_bytes = plugin.invoke(STATUS, Bytes::new()).expect("status");
    let status = RuntimeStatus::decode(&status_bytes).expect("decode status");
    assert_eq!(status.version, 1);
    if !status.pose_available {
        assert_eq!(
            plugin.invoke(START_SESSION, encoded_cfg),
            Err(Error::Unavailable)
        );
    }
}

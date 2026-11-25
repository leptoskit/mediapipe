# mediapipe

MediaPipe Tasks `v1.0.0` data models, camera preview overlay mapping, SHA-256 verified model pinning, dynamic native C-API runtime, and LeptosKit runtime plugin.

MediaPipe is a trademark of Google LLC. This project is independent software and is not affiliated with or endorsed by Google.

## Modules & Features

| Feature | What it provides |
|---|---|
| *(default, zero deps)* | `OverlayTransform` (`cover`, `contain`, `fill`, mirroring, quarter turns), MediaPipe v1.0.0 vision & audio task types (`PoseLandmarker*`, `ObjectDetector*`, `GestureRecognizer*`, `AudioClassifier*`, `PoseJoint`), `MpImage` pixel conversions (`BGRA8`, `NV21`, quarter-turn rotation), `TimestampTracker`, `FramePacer` |
| `serde` | `Serialize` / `Deserialize` on overlay & task types |
| `model-cache` | `ModelPin` & `ModelCache` enforcing immutable `storage.googleapis.com/mediapipe-models/` URLs, exact byte lengths, and SHA-256 verification on write and every read |
| `native` | Dynamic `libmediapipe.{so,dylib}` v1.0.0 C-API loader (`NativeRuntime`, `NativePoseLandmarker`, `NativeObjectDetector`, `NativeGestureRecognizer`, `NativeAudioClassifier`) matching `ifiokjr/mediapipe` `native-v1.0.0-1` archives; fails closed with `NativeError::RuntimeUnavailable` when absent |
| `leptoskit-guest` | `bridge::guest::GuestMediaPipe` Wasm client speaking the `"mediapipe"` `leptoskit_bridge` binary wire protocol |
| `leptoskit-host` | `bridge::HostMediaPipe` & `bridge::host_plugin` Rust host plugin + Android (`MediaPipePlugin.kt`) and iOS (`MediaPipePlugin.swift`) adapters declared under `[package.metadata.leptoskit.runtime]` |

## Development

```sh
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --locked
cargo test --all-features --locked
```

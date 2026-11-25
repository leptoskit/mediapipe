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

## Releases

Releases are managed with [monochange](https://crates.io/crates/monochange) changesets:

1. Add a changeset on your feature branch:

   ```sh
   monochange run change --package mediapipe --bump patch --reason "Describe the change"
   ```

2. Every pull request runs `changeset-policy` (enforcing changesets for crate edits and posting a change-classification summary) and `release pr (simulated)` (rehearsing version bumps, changelog updates, and `Cargo.lock` refresh without committing).
3. Pushing to `main` runs the `release` workflow, which uses the monochange GitHub App (`MONOCHANGE_RELEASE_BACKEND=hosted`) to open or refresh the single `chore(release): prepare release` pull request and dispatches a `publish.yml` dry run against it.
4. Merging the release pull request creates the `v<version>` tag, drafts the GitHub release with monochange's release notes, and dispatches `publish.yml` to validate, publish to crates.io (via trusted publishing or `CARGO_REGISTRY_TOKEN`), and flip the draft release public.

# mediapipe

Map [MediaPipe](https://developers.google.com/edge/mediapipe) vision results onto a camera preview.

Tasks report landmarks and boxes on the **upright** frame, after the rotation
they were given. A preview still has to crop, letterbox, or stretch that
frame, and a front camera is mirrored. Getting any one of those wrong puts
every keypoint in the wrong place. This crate does that mapping and nothing
else.

MediaPipe is a trademark of Google LLC. This project is independent software
and is not affiliated with or endorsed by Google.

## Usage

```rust
use mediapipe::{OverlayError, OverlayTransform, PreviewFit, Size};

fn place_landmark() -> Result<(), OverlayError> {
    let transform = OverlayTransform::new(
        Size::new(100.0, 100.0),
        0,
        Size::new(100.0, 100.0),
        true,
        PreviewFit::Cover,
    )?;
    let point = transform.map_normalized(0.25, 0.75);
    assert_eq!((point.x, point.y), (75.0, 75.0));
    Ok(())
}
```

`rotation_degrees` must be a multiple of 90, including negative and
past-360 values. Quarter turns swap the upright width and height. Points are
not rotated a second time.

`PreviewFit::Cover` crops around the centre, `Contain` letterboxes, and
`Fill` stretches each axis. Mirroring reflects about the upright image, so a
letterboxed front camera still mirrors the picture rather than the view.

## What is not in this crate

There is no native MediaPipe runtime, no model runner, and no platform camera
capture. Those pieces depend on Bazel builds and on macOS, iOS, and Android
toolchains that this Linux setup does not provide. The geometry here is the
part that can be tested without them.

## Development

The package pins Rust 1.83.0 in `rust-toolchain.toml`.

```sh
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
```

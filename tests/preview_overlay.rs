use mediapipe::{
    BoundingBox, NormalizedLandmark, NormalizedRect, OverlayError, OverlayTransform, Point,
    PreviewFit, Rect, Size,
};

fn transform(
    image: (f64, f64),
    rotation: i32,
    view: (f64, f64),
    mirrored: bool,
    fit: PreviewFit,
) -> OverlayTransform {
    OverlayTransform::new(
        Size::new(image.0, image.1),
        rotation,
        Size::new(view.0, view.1),
        mirrored,
        fit,
    )
    .unwrap_or_else(|error| panic!("transform should build: {error}"))
}

fn assert_close_point(actual: Point, x: f64, y: f64) {
    let distance = ((actual.x - x).powi(2) + (actual.y - y).powi(2)).sqrt();
    assert!(
        distance < 1e-6,
        "point ({}, {}) was not within 1e-6 of ({x}, {y})",
        actual.x,
        actual.y
    );
}

fn assert_close_rect(actual: Rect, left: f64, top: f64, right: f64, bottom: f64) {
    for (got, expected, edge) in [
        (actual.left, left, "left"),
        (actual.top, top, "top"),
        (actual.right, right, "right"),
        (actual.bottom, bottom, "bottom"),
    ] {
        assert!(
            (got - expected).abs() < 1e-9,
            "{edge} {got} was not within 1e-9 of {expected}"
        );
    }
}

#[test]
fn quarter_turns_swap_the_upright_size() {
    for rotation in [90, 270, -90, 450] {
        let mapped = transform(
            (640.0, 480.0),
            rotation,
            (100.0, 100.0),
            false,
            PreviewFit::Cover,
        );
        assert_eq!(
            mapped.upright_size(),
            Size::new(480.0, 640.0),
            "rotation {rotation}"
        );
    }
    for rotation in [0, 180, 360, -180] {
        let mapped = transform(
            (640.0, 480.0),
            rotation,
            (100.0, 100.0),
            false,
            PreviewFit::Cover,
        );
        assert_eq!(
            mapped.upright_size(),
            Size::new(640.0, 480.0),
            "rotation {rotation}"
        );
    }
}

#[test]
fn rotation_does_not_move_points_a_second_time() {
    let upright = transform((640.0, 480.0), 0, (320.0, 240.0), false, PreviewFit::Fill);
    let rotated = transform((640.0, 480.0), 180, (320.0, 240.0), false, PreviewFit::Fill);
    let point = upright.map_normalized(0.25, 0.1);
    assert_close_point(rotated.map_normalized(0.25, 0.1), point.x, point.y);
}

#[test]
fn rejects_invalid_sizes_and_rotations() {
    let image = Size::new(1.0, 1.0);
    let view = Size::new(1.0, 1.0);

    assert!(matches!(
        OverlayTransform::new(Size::new(0.0, 1.0), 0, view, false, PreviewFit::Cover),
        Err(OverlayError::NonPositiveImageSize { .. })
    ));
    assert!(matches!(
        OverlayTransform::new(Size::new(f64::NAN, 1.0), 0, view, false, PreviewFit::Cover),
        Err(OverlayError::NonPositiveImageSize { .. })
    ));
    assert!(matches!(
        OverlayTransform::new(
            Size::new(f64::INFINITY, 1.0),
            0,
            view,
            false,
            PreviewFit::Cover
        ),
        Err(OverlayError::NonPositiveImageSize { .. })
    ));
    assert!(matches!(
        OverlayTransform::new(image, 45, view, false, PreviewFit::Cover),
        Err(OverlayError::RotationNotMultipleOf90 { degrees: 45 })
    ));
    assert!(matches!(
        OverlayTransform::new(image, 0, Size::new(0.0, 1.0), false, PreviewFit::Cover),
        Err(OverlayError::NonPositiveViewSize { .. })
    ));
    assert!(matches!(
        OverlayTransform::new(
            image,
            0,
            Size::new(1.0, f64::NEG_INFINITY),
            false,
            PreviewFit::Cover
        ),
        Err(OverlayError::NonPositiveViewSize { .. })
    ));
}

#[test]
fn cover_crops_the_overflowing_axis_around_the_centre() {
    let mapped = transform((400.0, 300.0), 0, (200.0, 200.0), false, PreviewFit::Cover);

    assert_close_rect(mapped.source(), 50.0, 0.0, 350.0, 300.0);
    assert_close_rect(mapped.destination(), 0.0, 0.0, 200.0, 200.0);
    assert!((mapped.scale_x() - 2.0 / 3.0).abs() < 1e-9);
    assert!((mapped.scale_y() - 2.0 / 3.0).abs() < 1e-9);
    assert_close_point(mapped.map_normalized(0.5, 0.5), 100.0, 100.0);
    assert_close_point(mapped.map_normalized(0.0, 0.0), -100.0 / 3.0, 0.0);
    assert_close_point(mapped.map_normalized(1.0, 1.0), 200.0 + 100.0 / 3.0, 200.0);
}

#[test]
fn cover_crops_a_portrait_frame_vertically() {
    let mapped = transform((300.0, 400.0), 0, (200.0, 200.0), false, PreviewFit::Cover);

    assert_close_rect(mapped.source(), 0.0, 50.0, 300.0, 350.0);
    assert_close_point(mapped.map_normalized(0.5, 0.5), 100.0, 100.0);
}

#[test]
fn contain_letterboxes_and_keeps_the_whole_image_visible() {
    let mapped = transform(
        (400.0, 300.0),
        0,
        (200.0, 200.0),
        false,
        PreviewFit::Contain,
    );

    assert_close_rect(mapped.source(), 0.0, 0.0, 400.0, 300.0);
    assert_close_rect(mapped.destination(), 0.0, 25.0, 200.0, 175.0);
    assert_close_point(mapped.map_normalized(0.0, 0.0), 0.0, 25.0);
    assert_close_point(mapped.map_normalized(1.0, 1.0), 200.0, 175.0);
    assert_close_point(mapped.map_normalized(0.5, 0.5), 100.0, 100.0);
}

#[test]
fn contain_letterboxes_a_portrait_frame_horizontally() {
    let mapped = transform(
        (300.0, 400.0),
        0,
        (200.0, 200.0),
        false,
        PreviewFit::Contain,
    );

    assert_close_rect(mapped.destination(), 25.0, 0.0, 175.0, 200.0);
    assert_close_point(mapped.map_normalized(0.0, 0.0), 25.0, 0.0);
    assert_close_point(mapped.map_normalized(1.0, 1.0), 175.0, 200.0);
}

#[test]
fn fill_stretches_each_axis_independently() {
    let mapped = transform((400.0, 300.0), 0, (200.0, 600.0), false, PreviewFit::Fill);

    assert!((mapped.scale_x() - 0.5).abs() < 1e-9);
    assert!((mapped.scale_y() - 2.0).abs() < 1e-9);
    assert_close_point(mapped.map_pixel(400.0, 300.0), 200.0, 600.0);
}

#[test]
fn mirror_reflects_about_the_upright_image() {
    let mapped = transform((100.0, 100.0), 0, (100.0, 100.0), true, PreviewFit::Cover);

    assert_close_point(mapped.map_normalized(0.25, 0.75), 75.0, 75.0);
    assert_close_rect(
        mapped.map_bounding_box(BoundingBox::new(10.0, 20.0, 30.0, 40.0)),
        60.0,
        20.0,
        90.0,
        60.0,
    );
    assert_close_rect(
        mapped.map_normalized_rect(NormalizedRect::new(0.1, 0.2, 0.4, 0.5)),
        60.0,
        20.0,
        90.0,
        50.0,
    );
}

#[test]
fn mirror_stays_on_the_image_when_the_view_letterboxes() {
    let mapped = transform((400.0, 300.0), 0, (200.0, 200.0), true, PreviewFit::Contain);

    assert_close_point(mapped.map_normalized(0.0, 0.0), 200.0, 25.0);
    assert_close_point(mapped.map_normalized(1.0, 0.0), 0.0, 25.0);
}

#[test]
fn maps_landmarks_regions_and_boxes_after_rotation() {
    let mapped = transform(
        (480.0, 640.0),
        90,
        (320.0, 240.0),
        false,
        PreviewFit::Contain,
    );

    assert_eq!(mapped.upright_size(), Size::new(640.0, 480.0));
    assert_close_point(
        mapped.map_landmark(NormalizedLandmark::new(0.5, 0.25, 0.0)),
        160.0,
        60.0,
    );
    assert_close_rect(
        mapped.map_normalized_rect(NormalizedRect::new(0.25, 0.5, 0.75, 1.0)),
        80.0,
        120.0,
        240.0,
        240.0,
    );
    assert_close_rect(
        mapped.map_bounding_box(BoundingBox::new(64.0, 48.0, 128.0, 96.0)),
        32.0,
        24.0,
        96.0,
        72.0,
    );
}

#[test]
fn front_camera_quarter_turn_cover_places_the_origin_on_the_right() {
    let mapped = transform((4.0, 2.0), 270, (20.0, 40.0), true, PreviewFit::Cover);

    assert_eq!(mapped.upright_size(), Size::new(2.0, 4.0));
    assert!(mapped.mirrored());
    assert_eq!(mapped.fit(), PreviewFit::Cover);
    assert_close_point(mapped.map_normalized(0.0, 0.0), 20.0, 0.0);
}

//! Centered cover, contain, and fill placement of an upright camera frame.

/// Width and height in the same unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Size {
    /// Horizontal extent. Must be finite and positive for an image or a view.
    pub width: f64,
    /// Vertical extent. Must be finite and positive for an image or a view.
    pub height: f64,
}

impl Size {
    /// A size of `width` by `height`.
    #[must_use]
    pub const fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }

    fn is_positive(self) -> bool {
        self.width.is_finite() && self.height.is_finite() && self.width > 0.0 && self.height > 0.0
    }
}

/// A point in view or image space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Horizontal position.
    pub x: f64,
    /// Vertical position.
    pub y: f64,
}

impl Point {
    /// A point at `(x, y)`.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// An axis-aligned rectangle. `right` and `bottom` are exclusive of the
/// opposite edges only in the sense that width is `right - left`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Left edge.
    pub left: f64,
    /// Top edge.
    pub top: f64,
    /// Right edge.
    pub right: f64,
    /// Bottom edge.
    pub bottom: f64,
}

impl Rect {
    /// The rectangle spanning `a` and `b`, with edges ordered so `left <= right`
    /// and `top <= bottom`.
    #[must_use]
    pub fn from_points(a: Point, b: Point) -> Self {
        Self {
            left: a.x.min(b.x),
            top: a.y.min(b.y),
            right: a.x.max(b.x),
            bottom: a.y.max(b.y),
        }
    }

    /// `right - left`.
    #[must_use]
    pub fn width(self) -> f64 {
        self.right - self.left
    }

    /// `bottom - top`.
    #[must_use]
    pub fn height(self) -> f64 {
        self.bottom - self.top
    }
}

/// A landmark in normalized upright-image coordinates.
///
/// `x` and `y` are fractions of the upright frame, typically in `0.0..=1.0`.
/// `z` is the task's relative depth and is not a screen coordinate; mapping
/// ignores it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedLandmark {
    /// Horizontal position as a fraction of the upright width.
    pub x: f64,
    /// Vertical position as a fraction of the upright height.
    pub y: f64,
    /// Relative depth reported by the task. Not used when mapping to the view.
    pub z: f64,
}

impl NormalizedLandmark {
    /// A landmark at normalized `(x, y)` with relative depth `z`.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// A region in normalized upright-image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedRect {
    /// Left edge as a fraction of the upright width.
    pub left: f64,
    /// Top edge as a fraction of the upright height.
    pub top: f64,
    /// Right edge as a fraction of the upright width.
    pub right: f64,
    /// Bottom edge as a fraction of the upright height.
    pub bottom: f64,
}

impl NormalizedRect {
    /// A region from `(left, top)` to `(right, bottom)` in normalized coordinates.
    #[must_use]
    pub const fn new(left: f64, top: f64, right: f64, bottom: f64) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }
}

/// A pixel box on the upright image, origin at the top left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// Left edge in upright-image pixels.
    pub left: f64,
    /// Top edge in upright-image pixels.
    pub top: f64,
    /// Width in upright-image pixels.
    pub width: f64,
    /// Height in upright-image pixels.
    pub height: f64,
}

impl BoundingBox {
    /// A box whose top-left corner is `(left, top)`.
    #[must_use]
    pub const fn new(left: f64, top: f64, width: f64, height: f64) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }
}

/// How the upright frame is placed inside the view.
///
/// These match the three placements camera previews actually use:
/// crop to fill, letterbox, or stretch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewFit {
    /// Scale uniformly and crop the overflow. The view is filled.
    Cover,
    /// Scale uniformly and show the whole frame. The view may letterbox.
    Contain,
    /// Scale each axis independently so the frame fills the view.
    Fill,
}

/// Why a transform could not be built.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OverlayError {
    /// The sensor image has a non-finite or non-positive side.
    NonPositiveImageSize {
        /// Reported width.
        width: f64,
        /// Reported height.
        height: f64,
    },
    /// The view has a non-finite or non-positive side.
    NonPositiveViewSize {
        /// Reported width.
        width: f64,
        /// Reported height.
        height: f64,
    },
    /// Rotation was not a multiple of 90 degrees.
    RotationNotMultipleOf90 {
        /// Reported rotation in degrees.
        degrees: i32,
    },
}

impl std::fmt::Display for OverlayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonPositiveImageSize { width, height } => write!(
                f,
                "image size must be finite and positive, got {width}x{height}"
            ),
            Self::NonPositiveViewSize { width, height } => {
                write!(
                    f,
                    "view size must be finite and positive, got {width}x{height}"
                )
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

impl std::error::Error for OverlayError {}

/// Maps upright-image coordinates into a preview view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverlayTransform {
    upright_size: Size,
    view_size: Size,
    mirrored: bool,
    fit: PreviewFit,
    source: Rect,
    destination: Rect,
}

impl OverlayTransform {
    /// Build a transform for a sensor image of `image_size` pixels.
    ///
    /// `rotation_degrees` is the rotation applied before inference. It must be
    /// a multiple of 90. Quarter turns swap the upright width and height.
    /// `mirrored` reflects the preview horizontally about the upright image,
    /// which is what a front camera shows.
    ///
    /// # Errors
    ///
    /// Returns [`OverlayError`] when a size is non-finite or non-positive, or
    /// when `rotation_degrees` is not a multiple of 90.
    pub fn new(
        image_size: Size,
        rotation_degrees: i32,
        view_size: Size,
        mirrored: bool,
        fit: PreviewFit,
    ) -> Result<Self, OverlayError> {
        if !image_size.is_positive() {
            return Err(OverlayError::NonPositiveImageSize {
                width: image_size.width,
                height: image_size.height,
            });
        }
        if !view_size.is_positive() {
            return Err(OverlayError::NonPositiveViewSize {
                width: view_size.width,
                height: view_size.height,
            });
        }
        if rotation_degrees.rem_euclid(90) != 0 {
            return Err(OverlayError::RotationNotMultipleOf90 {
                degrees: rotation_degrees,
            });
        }

        let upright_size = upright_size(image_size, rotation_degrees);
        let fitted = apply_fit(fit, upright_size, view_size);
        let source = inscribe_center(fitted.source, upright_size);
        let destination = inscribe_center(fitted.destination, view_size);

        Ok(Self {
            upright_size,
            view_size,
            mirrored,
            fit,
            source,
            destination,
        })
    }

    /// Size of the image after the inference rotation, in pixels.
    #[must_use]
    pub const fn upright_size(self) -> Size {
        self.upright_size
    }

    /// Size of the view the preview is painted into.
    #[must_use]
    pub const fn view_size(self) -> Size {
        self.view_size
    }

    /// Whether the preview mirrors the upright image horizontally.
    #[must_use]
    pub const fn mirrored(self) -> bool {
        self.mirrored
    }

    /// How the upright image is fitted into the view.
    #[must_use]
    pub const fn fit(self) -> PreviewFit {
        self.fit
    }

    /// The part of the upright image that is visible, in image pixels.
    #[must_use]
    pub const fn source(self) -> Rect {
        self.source
    }

    /// Where [`Self::source`] lands in the view.
    #[must_use]
    pub const fn destination(self) -> Rect {
        self.destination
    }

    /// Horizontal scale from image pixels to view pixels.
    #[must_use]
    pub fn scale_x(self) -> f64 {
        self.destination.width() / self.source.width()
    }

    /// Vertical scale from image pixels to view pixels.
    #[must_use]
    pub fn scale_y(self) -> f64 {
        self.destination.height() / self.source.height()
    }

    /// View position of a point given in normalized upright-image coordinates.
    #[must_use]
    pub fn map_normalized(self, x: f64, y: f64) -> Point {
        self.map_pixel(x * self.upright_size.width, y * self.upright_size.height)
    }

    /// View position of a landmark. `z` is ignored.
    #[must_use]
    pub fn map_landmark(self, landmark: NormalizedLandmark) -> Point {
        self.map_normalized(landmark.x, landmark.y)
    }

    /// View position of a point given in upright-image pixels.
    #[must_use]
    pub fn map_pixel(self, x: f64, y: f64) -> Point {
        let sensor_x = if self.mirrored {
            self.upright_size.width - x
        } else {
            x
        };
        Point {
            x: self.destination.left + (sensor_x - self.source.left) * self.scale_x(),
            y: self.destination.top + (y - self.source.top) * self.scale_y(),
        }
    }

    /// View rectangle covering a normalized region.
    ///
    /// The result is ordered even when mirroring swaps the horizontal edges.
    #[must_use]
    pub fn map_normalized_rect(self, rect: NormalizedRect) -> Rect {
        Rect::from_points(
            self.map_normalized(rect.left, rect.top),
            self.map_normalized(rect.right, rect.bottom),
        )
    }

    /// View rectangle covering a pixel bounding box.
    ///
    /// The result is ordered even when mirroring swaps the horizontal edges.
    #[must_use]
    pub fn map_bounding_box(self, bounds: BoundingBox) -> Rect {
        Rect::from_points(
            self.map_pixel(bounds.left, bounds.top),
            self.map_pixel(bounds.left + bounds.width, bounds.top + bounds.height),
        )
    }
}

/// Fold `degrees` into `0..360`.
fn normalize_rotation(degrees: i32) -> i32 {
    degrees.rem_euclid(360)
}

fn upright_size(image_size: Size, rotation_degrees: i32) -> Size {
    let quarter_turns = normalize_rotation(rotation_degrees) / 90;
    if quarter_turns % 2 == 1 {
        Size::new(image_size.height, image_size.width)
    } else {
        image_size
    }
}

struct Fitted {
    source: Size,
    destination: Size,
}

/// Centered fit. Equal aspect ratios take the exact-fit branch of each mode.
fn apply_fit(fit: PreviewFit, input: Size, output: Size) -> Fitted {
    match fit {
        PreviewFit::Fill => Fitted {
            source: input,
            destination: output,
        },
        PreviewFit::Contain => {
            let destination = if output.width / output.height > input.width / input.height {
                Size::new(input.width * output.height / input.height, output.height)
            } else {
                Size::new(output.width, input.height * output.width / input.width)
            };
            Fitted {
                source: input,
                destination,
            }
        }
        PreviewFit::Cover => {
            let source = if output.width / output.height > input.width / input.height {
                Size::new(input.width, input.width * output.height / output.width)
            } else {
                Size::new(input.height * output.width / output.height, input.height)
            };
            Fitted {
                source,
                destination: output,
            }
        }
    }
}

fn inscribe_center(size: Size, into: Size) -> Rect {
    let left = (into.width - size.width) / 2.0;
    let top = (into.height - size.height) / 2.0;
    Rect {
        left,
        top,
        right: left + size.width,
        bottom: top + size.height,
    }
}

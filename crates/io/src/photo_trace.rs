//! Tracing marked regions in a photograph: the damage on a sheet-metal skin, outlined with a
//! marker (or any dark mark on a light ground), becomes closed polygons in pixel coordinates.
//!
//! The pipeline is deliberately simple and fully bounded: decode (PNG or JPEG, pure Rust, size
//! limits), threshold the luminance into a mask, despeckle it with a 3 × 3 majority vote,
//! label its 4-connected components, and walk the pixel-edge boundary of each component into
//! closed loops, keeping the outer loop (the largest). The loops are simplified with
//! Douglas–Peucker so a hand-drawn blob becomes a polygon of a few dozen points.
//!
//! Coordinates are pixels with y down (the image's own frame); the caller scales them to
//! millimetres and flips y. Nothing here reads files or knows about sketches.

use std::collections::HashMap;
use std::io::Cursor;

use image::{ImageFormat, ImageReader, Limits};
use solidcraft_geom::Vec2;

/// The most bytes a photo may have.
pub const MAX_PHOTO_BYTES: usize = 16 << 20;
/// The longest image edge, in pixels.
pub const MAX_PHOTO_EDGE: u32 = 4096;
/// The most regions a trace reports (the largest first).
pub const MAX_REGIONS: usize = 64;
const MAX_CODEC_BYTES: u64 = 256 << 20;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TraceError {
    #[error("{0}")]
    Image(String),
    #[error("{0}")]
    Invalid(String),
}

/// How a photo is read.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceOptions {
    /// Luminance below which a pixel is a mark (0 … 1 of white).
    pub threshold: f64,
    /// Marks are lighter than the ground (a silver marker on dark paint).
    pub invert: bool,
    /// Drop specks and fill pinholes (a 3 × 3 neighbour count) before labelling.
    pub despeckle: bool,
    /// Regions with fewer pixels than this are noise.
    pub min_area_px: usize,
    /// Douglas–Peucker tolerance for the outlines, in pixels.
    pub simplify_px: f64,
}

impl Default for TraceOptions {
    fn default() -> Self {
        TraceOptions { threshold: 0.5, invert: false, despeckle: true, min_area_px: 16, simplify_px: 1.0 }
    }
}

/// One marked region: its pixel count, bounding box and outer outline (a closed polygon of
/// pixel-edge corners, counter-clockwise once y is flipped to point up).
#[derive(Clone, Debug, PartialEq)]
pub struct TraceRegion {
    pub area_px: usize,
    /// `[min x, min y, max x, max y]` in pixels.
    pub bbox: [f64; 4],
    pub outline: Vec<Vec2>,
}

/// What a photo holds.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    pub width: u32,
    pub height: u32,
    /// Regions by pixel count, the largest first.
    pub regions: Vec<TraceRegion>,
}

/// The luminance of a PNG or JPEG, one byte per pixel, row-major from the top-left.
pub fn decode_luma(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), TraceError> {
    if bytes.is_empty() || bytes.len() > MAX_PHOTO_BYTES {
        return Err(TraceError::Invalid(format!("the photo must be 1 byte to {} MiB", MAX_PHOTO_BYTES >> 20)));
    }
    let format = image::guess_format(bytes).map_err(|_| TraceError::Invalid("the photo must be a PNG or JPEG".into()))?;
    if !matches!(format, ImageFormat::Png | ImageFormat::Jpeg) {
        return Err(TraceError::Invalid("the photo must be a PNG or JPEG".into()));
    }
    let decoded = std::panic::catch_unwind(|| {
        let mut reader = ImageReader::with_format(Cursor::new(bytes), format);
        let mut limits = Limits::default();
        limits.max_image_width = Some(MAX_PHOTO_EDGE);
        limits.max_image_height = Some(MAX_PHOTO_EDGE);
        limits.max_alloc = Some(MAX_CODEC_BYTES);
        reader.limits(limits);
        reader.decode().map(|img| img.into_luma8())
    })
    .map_err(|_| TraceError::Image("the image decoder failed".into()))?
    .map_err(|e| TraceError::Image(e.to_string()))?;
    let (w, h) = decoded.dimensions();
    if w == 0 || h == 0 || w > MAX_PHOTO_EDGE || h > MAX_PHOTO_EDGE {
        return Err(TraceError::Invalid(format!("the photo must be at most {MAX_PHOTO_EDGE} × {MAX_PHOTO_EDGE} pixels")));
    }
    let px = decoded.into_raw();
    if px.len() != (w as usize) * (h as usize) {
        return Err(TraceError::Image("the decoded image has the wrong size".into()));
    }
    Ok((w, h, px))
}

/// The marked regions of a photo.
pub fn trace(bytes: &[u8], o: &TraceOptions) -> Result<Trace, TraceError> {
    let (w, h, luma) = decode_luma(bytes)?;
    trace_luma(w, h, &luma, o)
}

/// The marked regions of a luminance image (`luma` is `w × h` bytes, row-major, top-left first).
pub fn trace_luma(w: u32, h: u32, luma: &[u8], o: &TraceOptions) -> Result<Trace, TraceError> {
    let (wi, hi) = (w as usize, h as usize);
    if wi == 0 || hi == 0 || w > MAX_PHOTO_EDGE || h > MAX_PHOTO_EDGE || luma.len() != wi * hi {
        return Err(TraceError::Invalid("the image size does not match its pixels".into()));
    }
    if !(o.threshold.is_finite() && (0.0..=1.0).contains(&o.threshold)) {
        return Err(TraceError::Invalid("`threshold` must be between 0 and 1".into()));
    }
    if !(o.simplify_px.is_finite() && o.simplify_px >= 0.0) {
        return Err(TraceError::Invalid("`simplify` must be zero or more".into()));
    }
    let cut = (o.threshold * 255.0).round() as i32;
    let mut mask: Vec<bool> = luma.iter().map(|&v| (i32::from(v) < cut) != o.invert).collect();
    if o.despeckle {
        mask = majority(wi, hi, &mask);
    }
    let regions = components(wi, hi, &mask, o.min_area_px.max(1));
    let mut out = Vec::new();
    for comp in regions.into_iter().take(MAX_REGIONS) {
        let Some(outline) = outer_loop(&comp.edges) else { continue };
        let mut outline = simplify_closed(&outline, o.simplify_px);
        // The walk runs clockwise once y points up; turn it round so callers that flip y get
        // a counter-clockwise polygon.
        outline.reverse();
        if outline.len() < 3 {
            continue;
        }
        out.push(TraceRegion { area_px: comp.area, bbox: comp.bbox, outline });
    }
    Ok(Trace { width: w, height: h, regions: out })
}

/// Specks go, pinholes fill, corners stay: a mark with fewer than two marked neighbours (of
/// eight) is dropped, a ground pixel with seven or more marked neighbours becomes a mark. Off
/// the image counts as ground.
fn majority(w: usize, h: usize, mask: &[bool]) -> Vec<bool> {
    let at = |x: isize, y: isize| -> bool {
        if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
            return false;
        }
        mask.get(y as usize * w + x as usize).copied().unwrap_or(false)
    };
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut n = 0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if (dx != 0 || dy != 0) && at(x as isize + dx, y as isize + dy) {
                        n += 1;
                    }
                }
            }
            let me = at(x as isize, y as isize);
            if let Some(o) = out.get_mut(y * w + x) {
                *o = if me { n >= 2 } else { n >= 7 };
            }
        }
    }
    out
}

/// A 4-connected component: its pixel count, box and boundary edges (pixel-edge segments with
/// the component on their right when y points down, so the loop runs clockwise on screen).
struct Component {
    area: usize,
    bbox: [f64; 4],
    edges: Vec<(Vec2, Vec2)>,
}

fn components(w: usize, h: usize, mask: &[bool], min_area: usize) -> Vec<Component> {
    let mut label = vec![0u32; w * h];
    let mut out: Vec<Component> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let is_set = |i: usize| mask.get(i).copied().unwrap_or(false);
    let mut next = 1u32;
    for start in 0..w * h {
        if !is_set(start) || label.get(start).copied().unwrap_or(1) != 0 {
            continue;
        }
        let id = next;
        next = next.saturating_add(1);
        stack.clear();
        stack.push(start);
        if let Some(l) = label.get_mut(start) {
            *l = id;
        }
        let mut pixels: Vec<usize> = Vec::new();
        while let Some(i) = stack.pop() {
            pixels.push(i);
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if is_set(j) && label.get(j).copied().unwrap_or(1) == 0 {
                    if let Some(l) = label.get_mut(j) {
                        *l = id;
                    }
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
        if pixels.len() < min_area {
            continue;
        }
        let mut bbox = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        let mut edges = Vec::new();
        let same = |j: usize| label.get(j).copied().unwrap_or(0) == id;
        for &i in &pixels {
            let (x, y) = (i % w, i / w);
            let (fx, fy) = (x as f64, y as f64);
            bbox = [bbox[0].min(fx), bbox[1].min(fy), bbox[2].max(fx + 1.0), bbox[3].max(fy + 1.0)];
            let (p00, p10, p11, p01) = (Vec2::new(fx, fy), Vec2::new(fx + 1.0, fy), Vec2::new(fx + 1.0, fy + 1.0), Vec2::new(fx, fy + 1.0));
            if y == 0 || !same(i - w) {
                edges.push((p00, p10));
            }
            if x + 1 >= w || !same(i + 1) {
                edges.push((p10, p11));
            }
            if y + 1 >= h || !same(i + w) {
                edges.push((p11, p01));
            }
            if x == 0 || !same(i - 1) {
                edges.push((p01, p00));
            }
        }
        out.push(Component { area: pixels.len(), bbox, edges });
    }
    out.sort_by(|a, b| b.area.cmp(&a.area));
    out
}

fn key(p: Vec2) -> (i64, i64) {
    (p.x.round() as i64, p.y.round() as i64)
}

/// The largest closed loop the boundary edges make (the outer one; the others bound holes).
/// Where two loops touch at a corner the walk turns toward the pixel it came along, so the
/// loops stay apart.
fn outer_loop(edges: &[(Vec2, Vec2)]) -> Option<Vec<Vec2>> {
    let mut from: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, (a, _)) in edges.iter().enumerate() {
        from.entry(key(*a)).or_default().push(i);
    }
    let mut used = vec![false; edges.len()];
    let mut best: Option<(f64, Vec<Vec2>)> = None;
    for start in 0..edges.len() {
        if used.get(start).copied().unwrap_or(true) {
            continue;
        }
        let mut lp: Vec<Vec2> = Vec::new();
        let mut cur = start;
        let mut steps = 0usize;
        loop {
            if let Some(u) = used.get_mut(cur) {
                *u = true;
            }
            let Some(&(a, b)) = edges.get(cur) else { break };
            lp.push(a);
            let d = b - a;
            let next = from.get(&key(b)).and_then(|cands| {
                let open: Vec<usize> = cands.iter().copied().filter(|&j| !used.get(j).copied().unwrap_or(true)).collect();
                match open.as_slice() {
                    [] => None,
                    [one] => Some(*one),
                    many => many.iter().copied().max_by(|&i, &j| {
                        let ti = edges.get(i).map_or(0.0, |(x, y)| d.cross(*y - *x));
                        let tj = edges.get(j).map_or(0.0, |(x, y)| d.cross(*y - *x));
                        ti.total_cmp(&tj)
                    }),
                }
            });
            steps += 1;
            match next {
                Some(n) if steps <= edges.len() => cur = n,
                _ => break,
            }
        }
        if lp.len() < 4 {
            continue;
        }
        let area = polygon_area(&lp).abs();
        if best.as_ref().is_none_or(|(a, _)| area > *a) {
            best = Some((area, lp));
        }
    }
    best.map(|(_, lp)| lp)
}

/// Twice the signed area of a closed polygon (shoelace).
pub fn polygon_area(p: &[Vec2]) -> f64 {
    let n = p.len();
    let mut a = 0.0;
    for i in 0..n {
        if let (Some(u), Some(v)) = (p.get(i), p.get((i + 1) % n)) {
            a += u.cross(*v);
        }
    }
    a / 2.0
}

fn dist_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f64 {
    let ab = b - a;
    let l2 = ab.len2();
    if l2 <= 1e-18 {
        return p.dist(a);
    }
    let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
    p.dist(a + ab * t)
}

/// Douglas–Peucker on an open polyline (both ends kept), iteratively.
fn simplify_open(pts: &[Vec2], tol: f64) -> Vec<Vec2> {
    let n = pts.len();
    if n < 3 {
        return pts.to_vec();
    }
    let mut keep = vec![false; n];
    if let Some(f) = keep.first_mut() {
        *f = true;
    }
    if let Some(l) = keep.last_mut() {
        *l = true;
    }
    let mut stack = vec![(0usize, n - 1)];
    while let Some((i, j)) = stack.pop() {
        if j <= i + 1 {
            continue;
        }
        let (Some(&a), Some(&b)) = (pts.get(i), pts.get(j)) else { continue };
        let mut far = (0.0f64, i);
        for k in i + 1..j {
            if let Some(&p) = pts.get(k) {
                let d = dist_to_segment(p, a, b);
                if d > far.0 {
                    far = (d, k);
                }
            }
        }
        if far.0 > tol && far.1 > i {
            if let Some(k) = keep.get_mut(far.1) {
                *k = true;
            }
            stack.push((i, far.1));
            stack.push((far.1, j));
        }
    }
    pts.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| *p).collect()
}

/// Douglas–Peucker on a closed polygon: it is cut at its first point and the point farthest
/// from it, each half simplified on its own.
pub fn simplify_closed(pts: &[Vec2], tol: f64) -> Vec<Vec2> {
    let n = pts.len();
    if n < 4 || tol <= 0.0 {
        return pts.to_vec();
    }
    let Some(&p0) = pts.first() else { return Vec::new() };
    let far = (1..n).max_by(|&i, &j| {
        let di = pts.get(i).map_or(0.0, |p| p.dist(p0));
        let dj = pts.get(j).map_or(0.0, |p| p.dist(p0));
        di.total_cmp(&dj)
    });
    let Some(far) = far else { return pts.to_vec() };
    let first: Vec<Vec2> = pts.get(..=far).map(simplify_open_slice(tol)).unwrap_or_default();
    let mut second: Vec<Vec2> = pts
        .get(far..)
        .map(|s| {
            let mut v = s.to_vec();
            v.push(p0);
            simplify_open(&v, tol)
        })
        .unwrap_or_default();
    // Both halves carry the cut points: drop the duplicates.
    let mut out = first;
    if second.len() > 1 {
        second.remove(0);
        second.pop();
        out.extend(second);
    }
    // Collinear corners the halves left behind (the cut points) are dropped too.
    let m = out.len();
    if m >= 4 {
        let mut kept = Vec::with_capacity(m);
        for i in 0..m {
            let (Some(&a), Some(&b), Some(&c)) = (out.get((i + m - 1) % m), out.get(i), out.get((i + 1) % m)) else { continue };
            if dist_to_segment(b, a, c) > tol || a.dist(c) < 1e-12 {
                kept.push(b);
            }
        }
        if kept.len() >= 3 {
            return kept;
        }
    }
    out
}

fn simplify_open_slice(tol: f64) -> impl Fn(&[Vec2]) -> Vec<Vec2> {
    move |s| simplify_open(s, tol)
}

/// A grey PNG of `w × h` bytes (row-major, top-left first), for tests and tools.
pub fn gray_png(w: u32, h: u32, luma: &[u8]) -> Option<Vec<u8>> {
    let img = image::GrayImage::from_raw(w, h, luma.to_vec())?;
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, ImageFormat::Png).ok()?;
    Some(buf.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A white `w × h` image with a black axis-aligned rectangle `[x0, x1) × [y0, y1)`.
    fn rect_image(w: u32, h: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> Vec<u8> {
        let mut v = vec![255u8; (w * h) as usize];
        for y in y0..y1 {
            for x in x0..x1 {
                v[(y * w + x) as usize] = 0;
            }
        }
        v
    }

    #[test]
    fn a_black_rectangle_traces_to_its_four_corners() {
        let luma = rect_image(40, 30, 5, 7, 25, 20);
        let t = trace_luma(40, 30, &luma, &TraceOptions::default()).unwrap();
        assert_eq!(t.regions.len(), 1);
        let r = &t.regions[0];
        assert_eq!(r.area_px, 20 * 13);
        assert_eq!(r.bbox, [5.0, 7.0, 25.0, 20.0]);
        let mut corners: Vec<(i64, i64)> = r.outline.iter().map(|p| key(*p)).collect();
        corners.sort();
        assert_eq!(corners, vec![(5, 7), (5, 20), (25, 7), (25, 20)]);
        // Counter-clockwise once y is up: negative in the image's y-down frame.
        assert!(polygon_area(&r.outline) < 0.0);
    }

    #[test]
    fn png_round_trip_and_speckles_are_dropped() {
        let mut luma = rect_image(60, 60, 10, 10, 40, 50);
        luma[3] = 0; // a lone dark pixel
        luma[59 * 60 + 58] = 0;
        let png = gray_png(60, 60, &luma).unwrap();
        let t = trace(&png, &TraceOptions::default()).unwrap();
        assert_eq!((t.width, t.height), (60, 60));
        assert_eq!(t.regions.len(), 1);
        assert_eq!(t.regions[0].area_px, 30 * 40);
    }

    #[test]
    fn inverted_marks_and_two_regions_sorted_by_size() {
        let mut luma = vec![0u8; 50 * 50];
        for y in 5..15 {
            for x in 5..30 {
                luma[y * 50 + x] = 255;
            }
        }
        for y in 30..36 {
            for x in 30..36 {
                luma[y * 50 + x] = 255;
            }
        }
        let t = trace_luma(50, 50, &luma, &TraceOptions { invert: true, ..Default::default() }).unwrap();
        assert_eq!(t.regions.len(), 2);
        assert_eq!(t.regions[0].area_px, 250);
        assert_eq!(t.regions[1].area_px, 36);
    }

    #[test]
    fn a_ring_keeps_only_its_outer_loop() {
        let mut luma = rect_image(40, 40, 5, 5, 35, 35);
        for y in 15..25 {
            for x in 15..25 {
                luma[y * 40 + x] = 255;
            }
        }
        let t = trace_luma(40, 40, &luma, &TraceOptions::default()).unwrap();
        assert_eq!(t.regions.len(), 1);
        assert_eq!(t.regions[0].outline.len(), 4);
        assert_eq!(t.regions[0].area_px, 900 - 100);
    }

    #[test]
    fn hostile_inputs_are_refused_not_panics() {
        assert!(trace(&[], &TraceOptions::default()).is_err());
        assert!(trace(b"not an image at all", &TraceOptions::default()).is_err());
        assert!(trace_luma(10, 10, &[0; 5], &TraceOptions::default()).is_err());
        assert!(trace_luma(0, 0, &[], &TraceOptions::default()).is_err());
        assert!(trace_luma(2, 2, &[0; 4], &TraceOptions { threshold: f64::NAN, ..Default::default() }).is_err());
        assert!(trace_luma(2, 2, &[0; 4], &TraceOptions { simplify_px: -1.0, ..Default::default() }).is_err());
        let truncated = &gray_png(8, 8, &[0; 64]).unwrap()[..20];
        assert!(trace(truncated, &TraceOptions::default()).is_err());
    }

    #[test]
    fn simplify_keeps_a_diamond() {
        let pts = vec![Vec2::new(0.0, 0.0), Vec2::new(5.0, 5.0), Vec2::new(10.0, 0.0), Vec2::new(5.0, -5.0)];
        assert_eq!(simplify_closed(&pts, 0.5), pts);
        let mut many = Vec::new();
        for i in 0..100 {
            let t = i as f64 / 100.0 * std::f64::consts::TAU;
            many.push(Vec2::new(t.cos() * 10.0, t.sin() * 10.0));
        }
        let few = simplify_closed(&many, 0.3);
        assert!(few.len() < 30 && few.len() >= 8, "{}", few.len());
    }
}

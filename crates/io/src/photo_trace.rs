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
/// The longest edge of the traced luminance grid, in pixels; larger photos are downsampled to it.
pub const MAX_PHOTO_EDGE: u32 = 4096;
/// The longest photo edge the decoder accepts at all.
pub const MAX_DECODE_EDGE: u32 = 12_000;
/// The most regions a trace reports (the largest first).
pub const MAX_REGIONS: usize = 512;
const MAX_CODEC_BYTES: u64 = 640 << 20;
/// The most boundary edges the kept regions may have between them (a 4096² image of 1-pixel
/// lines could otherwise cost gigabytes).
pub const MAX_BOUNDARY_EDGES: usize = 4_000_000;

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
    /// Regions with more pixels than this are not what is looked for (the panel, a shadow).
    pub max_area_px: usize,
    /// Douglas–Peucker tolerance for the outlines, in pixels.
    pub simplify_px: f64,
    /// Grow every mark by this many pixels all round before labelling (closes small gaps in
    /// a drawn line); 0 for none.
    pub grow_px: usize,
}

impl Default for TraceOptions {
    fn default() -> Self {
        TraceOptions { threshold: 0.5, invert: false, despeckle: true, min_area_px: 16, max_area_px: usize::MAX, simplify_px: 1.0, grow_px: 0 }
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

/// A decoded photo's luminance, downsampled when the photo is larger than [`MAX_PHOTO_EDGE`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Photo {
    /// The luminance grid's size (after any downsampling).
    pub width: u32,
    pub height: u32,
    /// One byte per pixel, row-major from the top-left.
    pub luma: Vec<u8>,
    /// Each luminance pixel covers `downsampled_by × downsampled_by` pixels of the photo
    /// (1: none). Coordinates in the photo's own pixels divide by this.
    pub downsampled_by: u32,
    /// The photo's own size in pixels.
    pub original: [u32; 2],
}

/// The luminance of a PNG or JPEG. A photo with an edge over [`MAX_PHOTO_EDGE`] pixels (a phone
/// camera's) is box-averaged down by the smallest whole factor that fits, so callers keep
/// working in the photo's own pixel coordinates (see [`Photo::downsampled_by`]).
pub fn decode_luma(bytes: &[u8]) -> Result<Photo, TraceError> {
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
        limits.max_image_width = Some(MAX_DECODE_EDGE);
        limits.max_image_height = Some(MAX_DECODE_EDGE);
        limits.max_alloc = Some(MAX_CODEC_BYTES);
        reader.limits(limits);
        reader.decode().map(|img| img.into_luma8())
    })
    .map_err(|_| TraceError::Image("the image decoder failed".into()))?
    .map_err(|e| TraceError::Image(e.to_string()))?;
    let (w, h) = decoded.dimensions();
    if w == 0 || h == 0 || w > MAX_DECODE_EDGE || h > MAX_DECODE_EDGE {
        return Err(TraceError::Invalid(format!("the photo must be at most {MAX_DECODE_EDGE} × {MAX_DECODE_EDGE} pixels")));
    }
    let px = decoded.into_raw();
    if px.len() != (w as usize) * (h as usize) {
        return Err(TraceError::Image("the decoded image has the wrong size".into()));
    }
    let f = w.max(h).div_ceil(MAX_PHOTO_EDGE).max(1);
    if f == 1 {
        return Ok(Photo { width: w, height: h, luma: px, downsampled_by: 1, original: [w, h] });
    }
    let (sw, sh) = (w.div_ceil(f), h.div_ceil(f));
    let (wi, fi) = (w as usize, f as usize);
    let mut luma = Vec::with_capacity((sw as usize) * (sh as usize));
    for sy in 0..sh as usize {
        for sx in 0..sw as usize {
            let (mut sum, mut n) = (0u32, 0u32);
            for y in sy * fi..((sy + 1) * fi).min(h as usize) {
                for x in sx * fi..((sx + 1) * fi).min(wi) {
                    sum += u32::from(px.get(y * wi + x).copied().unwrap_or(255));
                    n += 1;
                }
            }
            luma.push(sum.checked_div(n).map_or(255, |v| v as u8));
        }
    }
    Ok(Photo { width: sw, height: sh, luma, downsampled_by: f, original: [w, h] })
}

/// The marked regions of a photo (in the luminance grid's pixels: see [`Photo::downsampled_by`]).
pub fn trace(bytes: &[u8], o: &TraceOptions) -> Result<Trace, TraceError> {
    let photo = decode_luma(bytes)?;
    trace_luma(photo.width, photo.height, &photo.luma, o)
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
    if o.grow_px > 0 {
        mask = grow(wi, hi, &mask, o.grow_px.min(64));
    }
    let regions = components(wi, hi, &mask, o.min_area_px.max(1), o.max_area_px)?;
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

/// Every mark grown by `r` pixels along both axes (a square dilation, separable).
fn grow(w: usize, h: usize, mask: &[bool], r: usize) -> Vec<bool> {
    let mut rows = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            if (lo..=hi).any(|i| mask.get(y * w + i).copied().unwrap_or(false))
                && let Some(o) = rows.get_mut(y * w + x)
            {
                *o = true;
            }
        }
    }
    let mut out = vec![false; w * h];
    for y in 0..h {
        let lo = y.saturating_sub(r);
        let hi = (y + r).min(h - 1);
        for x in 0..w {
            if (lo..=hi).any(|j| rows.get(j * w + x).copied().unwrap_or(false))
                && let Some(o) = out.get_mut(y * w + x)
            {
                *o = true;
            }
        }
    }
    out
}

/// The median luminance (0 … 1 of white) of an image.
pub fn median_luma(luma: &[u8]) -> f64 {
    let mut hist = [0usize; 256];
    for &v in luma {
        hist[usize::from(v)] += 1;
    }
    let half = luma.len() / 2;
    let mut seen = 0usize;
    hist.iter()
        .position(|&n| {
            seen += n;
            seen > half
        })
        .unwrap_or(128) as f64
        / 255.0
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

fn components(w: usize, h: usize, mask: &[bool], min_area: usize, max_area: usize) -> Result<Vec<Component>, TraceError> {
    let mut label = vec![0u32; w * h];
    let mut stack: Vec<usize> = Vec::new();
    let is_set = |i: usize| mask.get(i).copied().unwrap_or(false);
    // Pass 1: label every component and keep only the largest `MAX_REGIONS` (by pixel count),
    // so a field of specks never costs more than its label map.
    let mut kept: Vec<(usize, u32)> = Vec::new();
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
        let mut area = 0usize;
        while let Some(i) = stack.pop() {
            area += 1;
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
        if area < min_area || area > max_area {
            continue;
        }
        kept.push((area, id));
        if kept.len() >= 2 * MAX_REGIONS {
            kept.sort_by_key(|&(area, _)| std::cmp::Reverse(area));
            kept.truncate(MAX_REGIONS);
        }
    }
    kept.sort_by_key(|&(area, _)| std::cmp::Reverse(area));
    kept.truncate(MAX_REGIONS);
    let slot: HashMap<u32, usize> = kept.iter().enumerate().map(|(k, &(_, id))| (id, k)).collect();
    let mut out: Vec<Component> = kept
        .iter()
        .map(|&(area, _)| Component { area, bbox: [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], edges: Vec::new() })
        .collect();
    // Pass 2: the boxes and boundary edges of the kept components, in one sweep; the edge total
    // is capped so a maze-like mark cannot exhaust memory.
    let mut edges_total = 0usize;
    for i in 0..w * h {
        let id = label.get(i).copied().unwrap_or(0);
        if id == 0 {
            continue;
        }
        let Some(&k) = slot.get(&id) else { continue };
        let Some(comp) = out.get_mut(k) else { continue };
        let (x, y) = (i % w, i / w);
        let (fx, fy) = (x as f64, y as f64);
        comp.bbox = [comp.bbox[0].min(fx), comp.bbox[1].min(fy), comp.bbox[2].max(fx + 1.0), comp.bbox[3].max(fy + 1.0)];
        let same = |j: usize| label.get(j).copied().unwrap_or(0) == id;
        let (p00, p10, p11, p01) = (Vec2::new(fx, fy), Vec2::new(fx + 1.0, fy), Vec2::new(fx + 1.0, fy + 1.0), Vec2::new(fx, fy + 1.0));
        let before = comp.edges.len();
        if y == 0 || !same(i - w) {
            comp.edges.push((p00, p10));
        }
        if x + 1 >= w || !same(i + 1) {
            comp.edges.push((p10, p11));
        }
        if y + 1 >= h || !same(i + w) {
            comp.edges.push((p11, p01));
        }
        if x == 0 || !same(i - 1) {
            comp.edges.push((p01, p00));
        }
        edges_total += comp.edges.len() - before;
        if edges_total > MAX_BOUNDARY_EDGES {
            return Err(TraceError::Invalid("the marks are too intricate to trace (raise `min_area` or turn on `despeckle`)".into()));
        }
    }
    Ok(out)
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

/// A small round mark: a rivet head, a hole, a dot.
#[derive(Clone, Debug, PartialEq)]
pub struct Blob {
    /// The centre of its bounding box, in the luminance grid's pixels.
    pub center: Vec2,
    /// The diameter of a disc of the same area, in pixels.
    pub diameter_px: f64,
    /// Darker than its surroundings (else lighter).
    pub dark: bool,
    pub area_px: usize,
}

/// What [`blobs`] looks for.
#[derive(Clone, Debug, PartialEq)]
pub struct BlobOptions {
    /// Luminance cut (0 … 1 of white) for both polarities; `None`: from the photo's median
    /// brightness (dark marks below half of it, light marks above halfway from it to white).
    pub threshold: Option<f64>,
    /// The diameters wanted, in pixels (inclusive).
    pub diameter_px: [f64; 2],
    /// Dark marks, light marks, or both.
    pub dark: bool,
    pub light: bool,
}

/// Round marks of the wanted size: regions whose box is nearly square and nearly filled
/// (a ring, a digit or a line is not one). Sorted by area, the largest first.
pub fn blobs(w: u32, h: u32, luma: &[u8], o: &BlobOptions) -> Result<Vec<Blob>, TraceError> {
    let [d0, d1] = o.diameter_px;
    if !(d0.is_finite() && d1.is_finite() && d0 > 0.0 && d1 >= d0) {
        return Err(TraceError::Invalid("the blob diameters must be positive, the smallest first".into()));
    }
    let area = |d: f64| std::f64::consts::PI * d * d / 4.0;
    // The photo's median brightness, for the adaptive cuts.
    let median = median_luma(luma);
    // What counts as dark around a light mark (the ring test).
    let dark_cut = (o.threshold.unwrap_or(median / 2.0) * 255.0).round() as i32;
    let mut out = Vec::new();
    for invert in [false, true] {
        if (invert && !o.light) || (!invert && !o.dark) {
            continue;
        }
        let threshold = match o.threshold {
            Some(t) => t,
            None if invert => median + (1.0 - median) / 2.0,
            None => median / 2.0,
        };
        if !(threshold.is_finite() && (0.0..=1.0).contains(&threshold)) {
            return Err(TraceError::Invalid("`threshold` must be between 0 and 1".into()));
        }
        let t = trace_luma(
            w,
            h,
            luma,
            &TraceOptions {
                threshold,
                invert,
                despeckle: true,
                min_area_px: (area(d0) * 0.5).floor().max(1.0) as usize,
                max_area_px: (area(d1) * 2.0).ceil() as usize,
                simplify_px: 0.0,
                grow_px: 0,
            },
        )?;
        for r in t.regions {
            let (bw, bh) = (r.bbox[2] - r.bbox[0], r.bbox[3] - r.bbox[1]);
            if bw <= 0.0 || bh <= 0.0 {
                continue;
            }
            let aspect = bw / bh;
            let fill = r.area_px as f64 / (bw * bh);
            let diameter_px = 2.0 * (r.area_px as f64 / std::f64::consts::PI).sqrt();
            if !(0.6..=1.67).contains(&aspect) || fill < 0.6 || diameter_px < d0 || diameter_px > d1 {
                continue;
            }
            let center = Vec2::new((r.bbox[0] + r.bbox[2]) / 2.0, (r.bbox[1] + r.bbox[3]) / 2.0);
            // The light inside of a drawn ring (a "0", a circled note) is round too: a light mark
            // whose surroundings are mostly dark is not a rivet head.
            if invert && ringed_by_dark(w, h, luma, center, diameter_px / 2.0, dark_cut) {
                continue;
            }
            out.push(Blob { center, diameter_px, dark: !invert, area_px: r.area_px });
        }
    }
    out.sort_by_key(|b| std::cmp::Reverse(b.area_px));
    Ok(out)
}

/// More than half of the pixels just outside a disc of radius `r` around `c` (a band 0.5 to
/// 3.5 pixels beyond it) are darker than `cut`: the disc is the inside of a drawn ring.
fn ringed_by_dark(w: u32, h: u32, luma: &[u8], c: Vec2, r: f64, cut: i32) -> bool {
    let (r0, r1) = (r + 0.5, r + 3.5);
    let (mut dark, mut n) = (0usize, 0usize);
    let (x0, x1) = (((c.x - r1).floor().max(0.0)) as u32, ((c.x + r1).ceil().max(0.0)) as u32);
    let (y0, y1) = (((c.y - r1).floor().max(0.0)) as u32, ((c.y + r1).ceil().max(0.0)) as u32);
    for y in y0..y1.min(h) {
        for x in x0..x1.min(w) {
            let d = Vec2::new(f64::from(x) + 0.5, f64::from(y) + 0.5).dist(c);
            if d < r0 || d > r1 {
                continue;
            }
            n += 1;
            if luma.get((y * w + x) as usize).is_some_and(|&v| i32::from(v) < cut) {
                dark += 1;
            }
        }
    }
    n > 0 && dark * 2 > n
}

/// The projective map (a 3 × 3 matrix, row-major) taking the four `from` points to the four
/// `to` points, in order; `None` when three of either are in line.
pub fn homography(from: [Vec2; 4], to: [Vec2; 4]) -> Option<[f64; 9]> {
    // Direct linear transform: eight equations in the eight unknowns of H with h33 = 1.
    let mut m = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let (p, q) = (from.get(i)?, to.get(i)?);
        if !(p.is_finite() && q.is_finite()) {
            return None;
        }
        *m.get_mut(2 * i)? = [p.x, p.y, 1.0, 0.0, 0.0, 0.0, -q.x * p.x, -q.x * p.y, q.x];
        *m.get_mut(2 * i + 1)? = [0.0, 0.0, 0.0, p.x, p.y, 1.0, -q.y * p.x, -q.y * p.y, q.y];
    }
    // Gaussian elimination with partial pivoting on the augmented 8 × 9 system.
    for col in 0..8 {
        let pivot = (col..8).max_by(|&a, &b| m.get(a).map_or(0.0, |r| r[col].abs()).total_cmp(&m.get(b).map_or(0.0, |r| r[col].abs())))?;
        if m.get(pivot)?[col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, pivot);
        let row = *m.get(col)?;
        for r in 0..8 {
            if r == col {
                continue;
            }
            let f = m.get(r)?[col] / row[col];
            if f != 0.0 {
                let target = m.get_mut(r)?;
                for c in 0..9 {
                    target[c] -= f * row[c];
                }
            }
        }
    }
    let mut h = [0.0; 9];
    for i in 0..8 {
        let r = m.get(i)?;
        h[i] = r[8] / r[i];
        if !h[i].is_finite() {
            return None;
        }
    }
    h[8] = 1.0;
    Some(h)
}

/// `p` through the homography `h`; `None` at its horizon.
pub fn map_point(h: &[f64; 9], p: Vec2) -> Option<Vec2> {
    let w = h[6] * p.x + h[7] * p.y + h[8];
    if w.abs() < 1e-12 {
        return None;
    }
    let q = Vec2::new((h[0] * p.x + h[1] * p.y + h[2]) / w, (h[3] * p.x + h[4] * p.y + h[5]) / w);
    q.is_finite().then_some(q)
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
    fn a_field_of_specks_keeps_only_the_largest_regions() {
        // A 512 × 512 checkerboard of single pixels, no despeckle, every speck kept: bounded
        // to MAX_REGIONS regions, plus one real mark that comes first.
        let n = 512usize;
        let mut luma = vec![255u8; n * n];
        for y in 0..n {
            for x in 0..n {
                if (x + y) % 2 == 0 {
                    luma[y * n + x] = 0;
                }
            }
        }
        for y in 100..140 {
            for x in 100..160 {
                luma[y * n + x] = 0;
            }
        }
        // No simplification either: a one-pixel speck would otherwise simplify away.
        let o = TraceOptions { despeckle: false, min_area_px: 1, simplify_px: 0.0, ..Default::default() };
        let t = trace_luma(n as u32, n as u32, &luma, &o).unwrap();
        assert_eq!(t.regions.len(), MAX_REGIONS);
        assert!(t.regions[0].area_px > 40 * 60);
        assert!(t.regions.iter().skip(1).all(|r| r.area_px == 1));
        // One serpentine 1-pixel line over the whole image (every other row, joined at
        // alternate ends) is refused, not traced: its boundary would be millions of edges.
        let n = 4096usize;
        let mut luma = vec![255u8; n * n];
        for y in 0..n {
            if y % 2 == 0 {
                for x in 0..n {
                    luma[y * n + x] = 0;
                }
            } else {
                luma[y * n + if y % 4 == 1 { n - 1 } else { 0 }] = 0;
            }
        }
        let err = trace_luma(n as u32, n as u32, &luma, &o).unwrap_err();
        assert!(err.to_string().contains("intricate"), "{err}");
    }

    #[test]
    fn a_large_photo_is_downsampled_and_keeps_its_marks() {
        // 8200 × 10 with a black block over columns 100..700: downsampled by 3 to 2734 × 4.
        let (w, h) = (8200u32, 10u32);
        let mut luma = vec![255u8; (w * h) as usize];
        for y in 0..h {
            for x in 100..700 {
                luma[(y * w + x) as usize] = 0;
            }
        }
        let photo = decode_luma(&gray_png(w, h, &luma).unwrap()).unwrap();
        assert_eq!((photo.downsampled_by, photo.original, photo.width, photo.height), (3, [8200, 10], 2734, 4));
        let t = trace_luma(photo.width, photo.height, &photo.luma, &TraceOptions { despeckle: false, ..Default::default() }).unwrap();
        assert_eq!(t.regions.len(), 1);
        // Columns 33 (two thirds black) to 232 inclusive, all four rows.
        assert_eq!(t.regions[0].area_px, 200 * 4);
        assert_eq!(t.regions[0].bbox, [33.0, 0.0, 233.0, 4.0]);
        // Within the limit nothing changes.
        let small = decode_luma(&gray_png(4096, 2, &[0; 8192]).unwrap()).unwrap();
        assert_eq!((small.downsampled_by, small.width), (1, 4096));
    }

    #[test]
    fn round_marks_are_found_and_rings_lines_and_digits_are_not() {
        let (w, h) = (300u32, 200u32);
        let mut luma = vec![200u8; (w * h) as usize];
        let disc = |luma: &mut Vec<u8>, cx: f64, cy: f64, r: f64, v: u8| {
            for y in 0..h {
                for x in 0..w {
                    if (f64::from(x) + 0.5 - cx).hypot(f64::from(y) + 0.5 - cy) <= r {
                        luma[(y * w + x) as usize] = v;
                    }
                }
            }
        };
        disc(&mut luma, 50.0, 50.0, 8.0, 20); // a dark hole
        disc(&mut luma, 120.0, 60.0, 8.0, 255); // a bright rivet head
        disc(&mut luma, 200.0, 50.0, 12.0, 20); // a ring (a drawn "0")
        disc(&mut luma, 200.0, 50.0, 8.0, 200);
        for x in 20..280 {
            luma[(150 * w + x) as usize] = 20; // a pen line
            luma[(151 * w + x) as usize] = 20;
        }
        disc(&mut luma, 250.0, 120.0, 30.0, 20); // far too big
        let o = BlobOptions { threshold: None, diameter_px: [8.0, 32.0], dark: true, light: true };
        let found = blobs(w, h, &luma, &o).unwrap();
        assert_eq!(found.len(), 2, "{found:?}");
        let hole = found.iter().find(|b| b.dark).unwrap();
        let head = found.iter().find(|b| !b.dark).unwrap();
        assert!(hole.center.dist(Vec2::new(50.0, 50.0)) < 0.6 && (hole.diameter_px - 16.0).abs() < 1.0, "{hole:?}");
        assert!(head.center.dist(Vec2::new(120.0, 60.0)) < 0.6 && (head.diameter_px - 16.0).abs() < 1.0, "{head:?}");
        let dark_only = blobs(w, h, &luma, &BlobOptions { light: false, ..o.clone() }).unwrap();
        assert_eq!(dark_only.len(), 1);
        assert!(blobs(w, h, &luma, &BlobOptions { diameter_px: [10.0, 5.0], ..o }).is_err());
    }

    #[test]
    fn a_homography_maps_its_corners_and_what_lies_between() {
        let from = [Vec2::new(100.0, 120.0), Vec2::new(1700.0, 80.0), Vec2::new(1760.0, 520.0), Vec2::new(60.0, 480.0)];
        let to = [Vec2::new(0.0, 0.0), Vec2::new(228.6, 0.0), Vec2::new(228.6, 50.8), Vec2::new(0.0, 50.8)];
        let h = homography(from, to).unwrap();
        for (p, q) in from.iter().zip(&to) {
            assert!(map_point(&h, *p).unwrap().dist(*q) < 1e-6);
        }
        // The inverse takes them back.
        let inv = homography(to, from).unwrap();
        let mid = map_point(&h, Vec2::new(900.0, 300.0)).unwrap();
        assert!(map_point(&inv, mid).unwrap().dist(Vec2::new(900.0, 300.0)) < 1e-6);
        // Three points in line have no homography.
        assert!(homography([Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0), Vec2::new(0.0, 1.0)], to).is_none());
    }

    #[test]
    fn growing_closes_a_gap_in_a_drawn_box() {
        // A 2-px pen rectangle with a 3-px gap in its top edge: ungrown it is a C whose outer
        // loop runs inside too; grown by 2 px it is a closed ring whose outer loop is the box.
        let (w, h) = (120u32, 80u32);
        let mut luma = vec![220u8; (w * h) as usize];
        for x in 20..100 {
            for y in [20u32, 21, 60, 61] {
                if y <= 21 && (55..58).contains(&x) {
                    continue;
                }
                luma[(y * w + x) as usize] = 20;
            }
        }
        for y in 20..62 {
            for x in [20u32, 21, 98, 99] {
                luma[(y * w + x) as usize] = 20;
            }
        }
        let o = TraceOptions { despeckle: false, simplify_px: 1.0, ..Default::default() };
        let open = trace_luma(w, h, &luma, &o).unwrap();
        assert!(polygon_area(&open.regions[0].outline).abs() < 80.0 * 42.0 * 0.5, "a C traces as a thin shape");
        let closed = trace_luma(w, h, &luma, &TraceOptions { grow_px: 2, ..o }).unwrap();
        let a = polygon_area(&closed.regions[0].outline).abs();
        assert!((a - 84.0 * 46.0).abs() < 84.0 * 46.0 * 0.03, "{a}");
        assert!((median_luma(&luma) - 220.0 / 255.0).abs() < 1e-9);
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

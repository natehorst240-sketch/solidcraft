//! Fitting geometry to the edges of a photograph: round openings (lightening holes) as circles
//! and the lower edge of a skin as an arc. Nothing is traced by colour; the photo's brightness
//! gradients are the evidence.
//!
//! Circles: a gradient-directed Hough transform (every strong edge pixel votes for the centres
//! that lie along its gradient at every candidate radius), peaks refined by a least-squares
//! circle through the edge pixels near them. The arc: in every column the lowest boundary with
//! light above and dark below (a window contrast), fitted with RANSAC so pen marks and the
//! bench below do not pull it.
//!
//! All coordinates are the luminance grid's pixels, y down.

use std::f64::consts::TAU;

use solidcraft_geom::Vec2;

use crate::photo_trace::TraceError;

/// A fitted circle.
#[derive(Clone, Debug, PartialEq)]
pub struct Circle {
    pub center: Vec2,
    pub radius: f64,
    /// Edge pixels within 4 px of the circle, per pixel of circumference (about 1 for a clean
    /// edge, more where the edge is wide; under 0.5 is doubtful).
    pub coverage: f64,
}

/// A fitted arc through the bottom edge.
#[derive(Clone, Debug, PartialEq)]
pub struct ArcFit {
    pub center: Vec2,
    pub radius: f64,
    /// The edge points that agree with the arc, and all of them.
    pub inliers: usize,
    pub points: usize,
    /// The columns the inliers span.
    pub x_range: [f64; 2],
}

/// Blurred and differentiated: the gradient of `luma` (two 3 × 3 binomial passes, Sobel).
struct Gradient {
    w: usize,
    h: usize,
    gx: Vec<f32>,
    gy: Vec<f32>,
    mag: Vec<f32>,
}

fn gradient(w: usize, h: usize, luma: &[u8]) -> Gradient {
    let mut g: Vec<f32> = luma.iter().map(|&v| f32::from(v)).collect();
    let at = |g: &[f32], x: usize, y: usize| g.get(y * w + x).copied().unwrap_or(0.0);
    for _ in 0..2 {
        let mut b = g.clone();
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let s = at(&g, x - 1, y - 1)
                    + 2.0 * at(&g, x, y - 1)
                    + at(&g, x + 1, y - 1)
                    + 2.0 * at(&g, x - 1, y)
                    + 4.0 * at(&g, x, y)
                    + 2.0 * at(&g, x + 1, y)
                    + at(&g, x - 1, y + 1)
                    + 2.0 * at(&g, x, y + 1)
                    + at(&g, x + 1, y + 1);
                if let Some(o) = b.get_mut(y * w + x) {
                    *o = s / 16.0;
                }
            }
        }
        g = b;
    }
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    let mut mag = vec![0.0f32; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let dx = (at(&g, x + 1, y - 1) + 2.0 * at(&g, x + 1, y) + at(&g, x + 1, y + 1))
                - (at(&g, x - 1, y - 1) + 2.0 * at(&g, x - 1, y) + at(&g, x - 1, y + 1));
            let dy = (at(&g, x - 1, y + 1) + 2.0 * at(&g, x, y + 1) + at(&g, x + 1, y + 1))
                - (at(&g, x - 1, y - 1) + 2.0 * at(&g, x, y - 1) + at(&g, x + 1, y - 1));
            let i = y * w + x;
            if let (Some(a), Some(b), Some(m)) = (gx.get_mut(i), gy.get_mut(i), mag.get_mut(i)) {
                *a = dx;
                *b = dy;
                *m = dx.hypot(dy);
            }
        }
    }
    Gradient { w, h, gx, gy, mag }
}

/// The magnitude below which `fraction` of the pixels lie.
fn percentile(mag: &[f32], fraction: f64) -> f32 {
    let max = mag.iter().copied().fold(0.0f32, f32::max);
    if max <= 0.0 {
        return 0.0;
    }
    let bins = 2048usize;
    let mut hist = vec![0usize; bins];
    for &m in mag {
        let b = ((m / max) as f64 * (bins - 1) as f64) as usize;
        if let Some(c) = hist.get_mut(b.min(bins - 1)) {
            *c += 1;
        }
    }
    let want = (mag.len() as f64 * fraction) as usize;
    let mut seen = 0usize;
    for (b, &c) in hist.iter().enumerate() {
        seen += c;
        if seen >= want {
            return max * (b as f32 / (bins - 1) as f32);
        }
    }
    max
}

/// An edge pixel and its unit gradient.
struct Edge {
    x: f32,
    y: f32,
    nx: f32,
    ny: f32,
}

fn edges(g: &Gradient, threshold: f32) -> Vec<Edge> {
    let mut out = Vec::new();
    for y in 0..g.h {
        for x in 0..g.w {
            let i = y * g.w + x;
            let m = g.mag.get(i).copied().unwrap_or(0.0);
            if m > threshold && m > 0.0 {
                out.push(Edge {
                    x: x as f32 + 0.5,
                    y: y as f32 + 0.5,
                    nx: g.gx.get(i).copied().unwrap_or(0.0) / m,
                    ny: g.gy.get(i).copied().unwrap_or(0.0) / m,
                });
            }
        }
    }
    out
}

/// Least-squares circle (Kåsa) through points; `None` when they are in line.
pub fn fit_circle(pts: &[Vec2]) -> Option<(Vec2, f64)> {
    if pts.len() < 3 {
        return None;
    }
    // Normal equations of  2x·a + 2y·b + c = x² + y²  (a, b the centre, c = r² − a² − b²).
    let (mut sxx, mut sxy, mut syy, mut sx, mut sy, mut sxz, mut syz, mut sz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let n = pts.len() as f64;
    for p in pts {
        let z = p.x * p.x + p.y * p.y;
        sxx += p.x * p.x;
        sxy += p.x * p.y;
        syy += p.y * p.y;
        sx += p.x;
        sy += p.y;
        sxz += p.x * z;
        syz += p.y * z;
        sz += z;
    }
    // [2sxx 2sxy sx; 2sxy 2syy sy; 2sx 2sy n] · [a b c]ᵀ = [sxz syz sz]ᵀ, solved by Cramer.
    let m = [[2.0 * sxx, 2.0 * sxy, sx], [2.0 * sxy, 2.0 * syy, sy], [2.0 * sx, 2.0 * sy, n]];
    let rhs = [sxz, syz, sz];
    let det3 = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det3(&m);
    if d.abs() < 1e-9 {
        return None;
    }
    let col = |k: usize| {
        let mut mm = m;
        for (row, r) in mm.iter_mut().enumerate() {
            r[k] = rhs[row];
        }
        det3(&mm) / d
    };
    let (a, b, c) = (col(0), col(1), col(2));
    let r2 = c + a * a + b * b;
    if !(r2.is_finite() && r2 > 0.0) {
        return None;
    }
    Some((Vec2::new(a, b), r2.sqrt()))
}

/// The `top` best circles with radii in `radius_px`, the strongest first.
pub fn circles(w: u32, h: u32, luma: &[u8], radius_px: [f64; 2], top: usize) -> Result<Vec<Circle>, TraceError> {
    let (wi, hi) = (w as usize, h as usize);
    if wi < 8 || hi < 8 || luma.len() != wi * hi {
        return Err(TraceError::Invalid("the image is too small to fit circles".into()));
    }
    let [r0, r1] = radius_px;
    if !(r0.is_finite() && r1.is_finite() && r0 >= 3.0 && r1 >= r0 && r1 < 1e5) {
        return Err(TraceError::Invalid("the circle radii must be at least 3 px, the smallest first".into()));
    }
    let g = gradient(wi, hi, luma);
    let es = edges(&g, percentile(&g.mag, 0.94));
    // Candidate radii: about 40 steps across the range, at least a pixel apart.
    let step_r = ((r1 - r0) / 40.0).max(1.0);
    let radii: Vec<f64> = std::iter::successors(Some(r0), |r| (r + step_r <= r1 + 1e-9).then_some(r + step_r)).collect();
    // The accumulator: centres on a 4 px grid.
    let cell = 4usize;
    let (aw, ah) = (wi / cell + 1, hi / cell + 1);
    let mut acc: Vec<Vec<u32>> = vec![vec![0u32; aw * ah]; radii.len()];
    for (ri, &r) in radii.iter().enumerate() {
        let Some(plane) = acc.get_mut(ri) else { continue };
        for e in &es {
            for sgn in [1.0f32, -1.0] {
                let cx = e.x + sgn * r as f32 * e.nx;
                let cy = e.y + sgn * r as f32 * e.ny;
                if cx < 0.0 || cy < 0.0 || cx >= wi as f32 || cy >= hi as f32 {
                    continue;
                }
                let i = (cy as usize / cell) * aw + cx as usize / cell;
                if let Some(v) = plane.get_mut(i) {
                    *v += 1;
                }
            }
        }
    }
    // Peaks, normalised by circumference, each suppressing its neighbourhood.
    let mut score: Vec<Vec<f64>> = acc.iter().zip(&radii).map(|(plane, &r)| plane.iter().map(|&v| f64::from(v) / (TAU * r)).collect()).collect();
    let mut out = Vec::new();
    for _ in 0..top.min(64) {
        let mut best = (0.0f64, 0usize, 0usize);
        for (ri, plane) in score.iter().enumerate() {
            for (i, &s) in plane.iter().enumerate() {
                if s > best.0 {
                    best = (s, ri, i);
                }
            }
        }
        if best.0 <= 0.0 {
            break;
        }
        let (_, ri, i) = best;
        let r = radii.get(ri).copied().unwrap_or(r0);
        let (cx, cy) = (((i % aw) * cell) as f64 + cell as f64 / 2.0, ((i / aw) * cell) as f64 + cell as f64 / 2.0);
        // Suppress everything within the radius of this centre, at every radius.
        let reach = (r / cell as f64).ceil() as usize + 1;
        let (gx0, gy0) = (i % aw, i / aw);
        for plane in score.iter_mut() {
            for yy in gy0.saturating_sub(reach)..(gy0 + reach + 1).min(ah) {
                for xx in gx0.saturating_sub(reach)..(gx0 + reach + 1).min(aw) {
                    if let Some(v) = plane.get_mut(yy * aw + xx) {
                        *v = 0.0;
                    }
                }
            }
        }
        // Refine on the edge pixels near the circle.
        let near: Vec<Vec2> = es
            .iter()
            .filter(|e| ((f64::from(e.x) - cx).hypot(f64::from(e.y) - cy) - r).abs() < 4.0)
            .map(|e| Vec2::new(f64::from(e.x), f64::from(e.y)))
            .collect();
        if near.len() < 30 {
            continue;
        }
        let Some((c, rr)) = fit_circle(&near) else { continue };
        if !(rr >= r0 * 0.8 && rr <= r1 * 1.25) {
            continue;
        }
        let coverage = near.len() as f64 / (TAU * rr);
        out.push(Circle { center: c, radius: rr, coverage });
    }
    out.sort_by(|a, b| b.coverage.total_cmp(&a.coverage));
    Ok(out)
}

/// In every other column from `from_y` down, the row with the strongest light-above /
/// dark-below contrast (10 rows above against 12 rows below, at least `min_contrast` apart).
pub fn bottom_edge_points(w: u32, h: u32, luma: &[u8], from_y: u32, min_contrast: f64) -> Vec<Vec2> {
    let (wi, hi) = (w as usize, h as usize);
    let mut out = Vec::new();
    if hi < 30 || luma.len() != wi * hi {
        return out;
    }
    let from = (from_y as usize + 10).min(hi);
    for x in (0..wi).step_by(2) {
        // Prefix sums down the column.
        let mut sum = Vec::with_capacity(hi + 1);
        sum.push(0u32);
        for y in 0..hi {
            let v = u32::from(luma.get(y * wi + x).copied().unwrap_or(0));
            sum.push(sum.last().copied().unwrap_or(0) + v);
        }
        let seg =
            |a: usize, b: usize| -> f64 { (sum.get(b).copied().unwrap_or(0) - sum.get(a).copied().unwrap_or(0)) as f64 / (b - a).max(1) as f64 };
        let mut best = (0.0f64, 0usize);
        for y in from..hi.saturating_sub(14) {
            let diff = seg(y - 10, y) - seg(y + 2, y + 14);
            if diff > best.0 {
                best = (diff, y);
            }
        }
        if best.0 >= min_contrast {
            out.push(Vec2::new(x as f64, best.1 as f64));
        }
    }
    out
}

/// RANSAC circle through the points (at least `min_radius_px`), the final fit on its inliers.
pub fn fit_arc(points: &[Vec2], tol_px: f64, min_radius_px: f64) -> Option<ArcFit> {
    if points.len() < 6 {
        return None;
    }
    let n = points.len();
    // A small deterministic generator (xorshift): the fit must be repeatable.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };
    let mut best: Option<(usize, Vec<bool>)> = None;
    for _ in 0..400 {
        let (i, j, k) = (next(), next(), next());
        if i == j || j == k || i == k {
            continue;
        }
        let (Some(&a), Some(&b), Some(&c)) = (points.get(i), points.get(j), points.get(k)) else { continue };
        let Some((cen, r)) = fit_circle(&[a, b, c]) else { continue };
        if r < min_radius_px {
            continue;
        }
        let inl: Vec<bool> = points.iter().map(|p| (p.dist(cen) - r).abs() < tol_px).collect();
        let count = inl.iter().filter(|&&b| b).count();
        if best.as_ref().is_none_or(|(bc, _)| count > *bc) {
            best = Some((count, inl));
        }
    }
    let (count, inl) = best?;
    if count < 6 {
        return None;
    }
    let sel: Vec<Vec2> = points.iter().zip(&inl).filter(|(_, b)| **b).map(|(p, _)| *p).collect();
    let (center, radius) = fit_circle(&sel)?;
    let (x0, x1) = sel.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| (a.min(p.x), b.max(p.x)));
    Some(ArcFit { center, radius, inliers: count, points: n, x_range: [x0, x1] })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1600 × 900: light skin above an arc (centre (800, −1200), radius 1900 px), dark below,
    /// two dark discs in the skin, pen scribble and a stray dark rectangle below the edge.
    pub(crate) fn skin_image() -> (u32, u32, Vec<u8>) {
        let (w, h) = (1600u32, 900u32);
        let mut luma = vec![190u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let p = Vec2::new(f64::from(x) + 0.5, f64::from(y) + 0.5);
                let i = (y * w + x) as usize;
                if p.dist(Vec2::new(800.0, -1200.0)) > 1900.0 {
                    luma[i] = 40;
                }
                if p.dist(Vec2::new(400.0, 300.0)) <= 100.0 || p.dist(Vec2::new(1200.0, 320.0)) <= 90.0 {
                    luma[i] = 30;
                }
                if (y == 450 || y == 451) && x > 600 && x < 1000 {
                    luma[i] = 20; // a pen line
                }
                if (700..760).contains(&y) && (300..500).contains(&x) {
                    luma[i] = 230; // something light on the bench below the edge
                }
            }
        }
        (w, h, luma)
    }

    #[test]
    fn two_discs_are_found_with_their_radii() {
        let (w, h, luma) = skin_image();
        let found = circles(w, h, &luma, [60.0, 140.0], 4).unwrap();
        assert!(found.len() >= 2, "{found:?}");
        let near = |c: Vec2, r: f64| found.iter().any(|f| f.center.dist(c) < 3.0 && (f.radius - r).abs() < 2.5 && f.coverage > 0.8);
        assert!(near(Vec2::new(400.0, 300.0), 100.0), "{found:?}");
        assert!(near(Vec2::new(1200.0, 320.0), 90.0), "{found:?}");
        assert!(circles(w, h, &luma, [10.0, 5.0], 2).is_err());
    }

    #[test]
    fn the_bottom_edge_fits_its_arc_despite_the_bench() {
        let (w, h, luma) = skin_image();
        let pts = bottom_edge_points(w, h, &luma, 420, 45.0);
        assert!(pts.len() > 500, "{}", pts.len());
        let arc = fit_arc(&pts, 3.0, 300.0).unwrap();
        assert!((arc.radius - 1900.0).abs() < 20.0, "{arc:?}");
        assert!(arc.center.dist(Vec2::new(800.0, -1200.0)) < 20.0, "{arc:?}");
        assert!(arc.inliers as f64 > 0.8 * arc.points as f64, "{arc:?}");
        assert!(arc.x_range[0] < 10.0 && arc.x_range[1] > 1580.0);
        assert!(fit_arc(&pts[..3], 3.0, 300.0).is_none());
    }

    #[test]
    fn a_circle_through_three_points() {
        let (c, r) = fit_circle(&[Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(-1.0, 0.0)]).unwrap();
        assert!(c.dist(Vec2::ZERO) < 1e-9 && (r - 1.0).abs() < 1e-9);
        assert!(fit_circle(&[Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0)]).is_none());
    }
}

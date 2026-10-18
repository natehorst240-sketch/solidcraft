//! Sheet-metal repair doublers: the damage on a skin (a polygon, or traced from a photograph
//! of the marked-up skin) becomes a clean-out cutout, a doubler plate that covers it with the
//! rivet rows around it, and a DXF ready for cutting — or a sketch to carry on with in
//! SolidCraft (`sheet.flange` on it makes the plate, `sheet.export_dxf` its flat pattern).
//!
//! The layout follows general practice for riveted patch repairs (for instance FAA AC 43.13-1B,
//! chapter 4): the damage is trimmed back to a smooth shape (a circle or a rectangle with
//! rounded corners) with a margin, the first rivet row sits one edge distance outside that
//! cutout, further rows one row pitch further out, and the doubler's edge is one edge distance
//! outside the last row. The defaults (edge distance 2.5 D, pitch 5 D, row pitch 0.75 of the
//! pitch, two rows) are starting values; the applicable repair data decides the real ones.

use std::f64::consts::{FRAC_PI_2, PI};

use serde_json::{Value, json};
use solidcraft_doc::canvas::base64_decode;
use solidcraft_geom::{Loop2, Seg2, Vec2};
use solidcraft_io::photo_trace::{self, MAX_PHOTO_BYTES, TraceOptions};
use solidcraft_io::{DxfEntity, DxfLayer, DxfShape};

use super::CommandSpec;
use super::sketch::edit;
use crate::params::{bad, bool_, num, str_, vec2};
use crate::{EngineError, Result, Session};

pub static COMMANDS: &[CommandSpec] = &[
    CommandSpec::new("repair.trace_photo", "Trace Damage from Photo", trace_photo).at("SHEET METAL", "CREATE").icon("canvas").noundo().doc_only().params(
        "path | data (base64 PNG or JPEG of the skin with the damage outlined or filled in with a marker); threshold?: 0…1 of white (default 0.5), invert?: bool (light marks on a dark skin), despeckle?: bool (default true); \
         scale: mm_per_pixel | width_mm (the photo's real width) | calibrate: {a: [px,py], b: [px,py], distance: mm} (two points a known distance apart); \
         min_area?: mm² (marks smaller are noise, default 1); simplify?: mm (outline tolerance, default 0.5); \
         origin?: center (default: the chosen mark's centre is 0,0) | image (the photo's bottom-left) | [px,py]; region?: index | at?: [px,py] (which mark is `chosen`) \
         (pixel coordinates are always the photo's own; a photo over 4096 px on an edge is traced on a coarser grid, downsampled_by) \
         → {width_px, height_px, mm_per_pixel, downsampled_by, chosen, regions: [{area_mm2, bbox, outline: [[x,y], …] mm, y up, counter-clockwise; bbox_px, center_px: where the mark is in the photo's pixels}]}",
    ),
    CommandSpec::new("repair.doubler", "Repair Doubler", doubler).at("SHEET METAL", "CREATE").icon("flat").noundo().doc_only().params(
        "damage: [[x,y], …] mm (the damaged area's outline, 3 or more points) | photo: {the parameters of repair.trace_photo; region? or at? picks the mark, default the largest}; \
         thickness?: mm (the skin and doubler, default 1); rivet_diameter?: mm (default 3.2); edge_distance?: mm (default 2.5 × diameter); \
         pitch?: mm (along a row, default 5 × diameter); row_pitch?: mm (between rows, default 0.75 × pitch); rows?: 1…10 (default 2); stagger?: bool (default true); \
         cutout?: rectangle|circle (the clean-out shape, default rectangle: the smallest around the damage, corners rounded); cleanup_margin?: mm beyond the damage (default 3); \
         corner_radius?: mm (default 6); angle?: deg (the rectangle's direction, default the tightest fit); \
         path?: DXF file to write (else the DXF text is returned), notes?: bool (a note on layer NOTES, default true) \
         → {cutout, doubler: {type: circle|rounded_rectangle, center, angle_deg, width, height, radius, segments}, rivets: [[x,y], …], rivet_diameter, rows: [{offset, count, pitch}], damage: {points, area}, dxf | path}; \
         DXF layers: DOUBLER (the plate's outline), RIVET (its holes), CUTOUT (dashed: what to trim out of the skin), DAMAGE (dashed: the damage as given), NOTES"
    ),
    CommandSpec::new("repair.skin_from_photo", "Skin from Photo", skin_from_photo).at("SHEET METAL", "CREATE").icon("flat").params(
        "the flat model of a skin from a photo: path | data; the scale as repair.trace_photo takes it (calibrate: {a, b, distance} | mm_per_pixel | width_mm); \
         outline: [[px,py], …] (3 or more corners of the region to model, in the photo's pixels; default the whole photo); \
         rectify?: {corners: [[px,py] × 4 top-left, top-right, bottom-right, bottom-left], width_mm, height_mm} (a rectangle of known size in the photo: \
         the photo is straightened through it and it sets the scale; outline defaults to it); \
         holes?: bool (default true: find the holes and rivet heads), hole_diameter?: mm (the one to expect, default 3.2; marks from half to twice it are taken), \
         polarity?: dark|light|both (default both), threshold?: 0…1 of white (default from the photo's brightness), drawn_diameter?: mm (draw every hole this size, default as measured); \
         thickness?: mm (a sheet rule `Skin` of this thickness becomes active; default the active rule), body?: bool (default true: sheet.flange on the sketch), \
         name? (default Skin), plane? (default XY) \
         → {sketch, body, outline: [[x,y], …] mm, size: [w, h], holes: [{center: [x,y], diameter, center_px, dark}], mm_per_pixel}; \
         the origin is the outline's bottom-left corner, y up",
    ),
    CommandSpec::new("repair.doubler_sketch", "Repair Doubler Sketch", doubler_sketch).at("SHEET METAL", "CREATE").icon("flat").params(
        "the parameters of repair.doubler, drawn as a sketch (the plate's outline and its rivet holes): plane? (when no sketch is active, a new sketch on it; default XY), name? (default Doubler), at?: [x,y] offset; \
         sheet.flange {sketch, profiles: [{point: profile_point}]} makes the plate with its holes, sheet.export_dxf its flat pattern → {sketch, created_sketch, curves, rivets, size, profile_point: [x,y] on the plate}"
    ),
];

/// The clean-out, a rivet row's line or the doubler's edge: a circle or a rectangle with
/// rounded corners, in a frame at `center` turned `angle`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Shape {
    center: Vec2,
    angle: f64,
    kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Circle {
        r: f64,
    },
    /// Half sizes `a` (along the frame's x) and `b`, corners of radius `r` (at most the smaller half size).
    Rounded {
        a: f64,
        b: f64,
        r: f64,
    },
}

impl Shape {
    /// The same shape `d` further out everywhere.
    fn offset(self, d: f64) -> Shape {
        let kind = match self.kind {
            Kind::Circle { r } => Kind::Circle { r: r + d },
            Kind::Rounded { a, b, r } => Kind::Rounded { a: a + d, b: b + d, r: r + d },
        };
        Shape { kind, ..self }
    }

    /// A point in the shape's own frame.
    fn to_local(self, p: Vec2) -> Vec2 {
        let (s, c) = self.angle.sin_cos();
        let q = p - self.center;
        Vec2::new(q.x * c + q.y * s, -q.x * s + q.y * c)
    }

    /// Grows a rounded rectangle, about its centre, until every point is inside it: a rounded
    /// rectangle is the inner rectangle (half sizes `a − r`, `b − r`) swept by a disc of `r`, so
    /// a point is inside when it is within `r` of that inner rectangle. Circles are unchanged.
    fn grow_to_contain(&mut self, pts: &[Vec2]) {
        let Kind::Rounded { a, b, r } = &mut self.kind else { return };
        let (center, angle) = (self.center, self.angle);
        let local = |p: Vec2| Shape { center, angle, kind: Kind::Circle { r: 0.0 } }.to_local(p);
        for _ in 0..3 {
            let (ia, ib) = ((*a - *r).max(0.0), (*b - *r).max(0.0));
            let outside = pts
                .iter()
                .map(|&p| {
                    let q = local(p);
                    let (dx, dy) = ((q.x.abs() - ia).max(0.0), (q.y.abs() - ib).max(0.0));
                    (dx * dx + dy * dy).sqrt() - *r
                })
                .fold(0.0, f64::max);
            if outside <= 1e-9 {
                break;
            }
            *a += outside;
            *b += outside;
        }
    }

    fn size(self) -> (f64, f64) {
        match self.kind {
            Kind::Circle { r } => (2.0 * r, 2.0 * r),
            Kind::Rounded { a, b, .. } => (2.0 * a, 2.0 * b),
        }
    }

    fn to_world(self, p: Vec2) -> Vec2 {
        let (s, c) = self.angle.sin_cos();
        self.center + Vec2::new(p.x * c - p.y * s, p.x * s + p.y * c)
    }

    /// The outline, counter-clockwise.
    fn outline(self) -> Loop2 {
        match self.kind {
            Kind::Circle { r } => Loop2::circle(self.center, r),
            Kind::Rounded { a, b, r } => {
                let r = r.min(a).min(b).max(0.0);
                let mut segs = Vec::new();
                let corners = [
                    (Vec2::new(-a + r, -b), Vec2::new(a - r, -b), Vec2::new(a - r, -b + r), -FRAC_PI_2),
                    (Vec2::new(a, -b + r), Vec2::new(a, b - r), Vec2::new(a - r, b - r), 0.0),
                    (Vec2::new(a - r, b), Vec2::new(-a + r, b), Vec2::new(-a + r, b - r), FRAC_PI_2),
                    (Vec2::new(-a, b - r), Vec2::new(-a, -b + r), Vec2::new(-a + r, -b + r), PI),
                ];
                for (p, q, c, start) in corners {
                    if p.dist(q) > 1e-9 {
                        segs.push(Seg2::Line { a: self.to_world(p), b: self.to_world(q) });
                    }
                    if r > 1e-9 {
                        segs.push(Seg2::Arc { center: self.to_world(c), radius: r, start: start + self.angle, sweep: FRAC_PI_2 });
                    }
                }
                Loop2 { segs }
            }
        }
    }

    fn json(self) -> Value {
        let (w, h) = self.size();
        let lp = self.outline();
        let segments: Vec<Value> = lp
            .segs
            .iter()
            .map(|s| match *s {
                Seg2::Line { a, b } => json!({"line": [pt(a), pt(b)]}),
                Seg2::Arc { center, radius, start, sweep } => {
                    json!({"arc": {"center": pt(center), "radius": radius, "start_deg": start.to_degrees(), "sweep_deg": sweep.to_degrees()}})
                }
                _ => Value::Null,
            })
            .collect();
        match self.kind {
            Kind::Circle { r } => {
                json!({"type": "circle", "center": pt(self.center), "radius": r, "width": w, "height": h, "angle_deg": 0.0, "segments": segments})
            }
            Kind::Rounded { r, .. } => {
                json!({"type": "rounded_rectangle", "center": pt(self.center), "angle_deg": self.angle.to_degrees(), "width": w, "height": h, "radius": r.min(w / 2.0).min(h / 2.0), "segments": segments})
            }
        }
    }

    fn dxf(self, layer: &str) -> Vec<DxfEntity> {
        match self.kind {
            Kind::Circle { r } => vec![DxfEntity { layer: layer.to_string(), shape: DxfShape::Circle { c: self.center, r } }],
            Kind::Rounded { .. } => {
                self.outline().segs.iter().map(|s| DxfEntity { layer: layer.to_string(), shape: solidcraft_io::seg_shape(s) }).collect()
            }
        }
    }
}

fn pt(p: Vec2) -> Value {
    json!([p.x, p.y])
}

/// The convex hull (Andrew's monotone chain), counter-clockwise, without repeated points.
fn hull(pts: &[Vec2]) -> Vec<Vec2> {
    let mut p: Vec<Vec2> = pts.to_vec();
    p.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    p.dedup_by(|a, b| a.dist(*b) < 1e-9);
    if p.len() < 3 {
        return p;
    }
    let mut lower: Vec<Vec2> = Vec::new();
    for &q in &p {
        while lower.len() >= 2 && lower.get(lower.len() - 2).zip(lower.last()).is_some_and(|(a, b)| (*b - *a).cross(q - *a) <= 1e-12) {
            lower.pop();
        }
        lower.push(q);
    }
    let mut upper: Vec<Vec2> = Vec::new();
    for &q in p.iter().rev() {
        while upper.len() >= 2 && upper.get(upper.len() - 2).zip(upper.last()).is_some_and(|(a, b)| (*b - *a).cross(q - *a) <= 1e-12) {
            upper.pop();
        }
        upper.push(q);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// The smallest-area rectangle around the hull (one side along a hull edge), or at `angle`.
fn min_rect(h: &[Vec2], angle: Option<f64>) -> Shape {
    let n = h.len();
    let dirs: Vec<f64> = match angle {
        Some(a) => vec![a],
        None => (0..n).filter_map(|i| h.get(i).zip(h.get((i + 1) % n))).filter_map(|(a, b)| (*b - *a).normalized().map(|d| d.angle())).collect(),
    };
    let mut best: Option<(f64, Shape)> = None;
    for th in dirs.into_iter().chain(std::iter::once(0.0)) {
        let (s, c) = th.sin_cos();
        let (u, v) = (Vec2::new(c, s), Vec2::new(-s, c));
        let (mut u0, mut u1, mut v0, mut v1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for &p in h {
            u0 = u0.min(p.dot(u));
            u1 = u1.max(p.dot(u));
            v0 = v0.min(p.dot(v));
            v1 = v1.max(p.dot(v));
        }
        let area = (u1 - u0) * (v1 - v0);
        let center = u * ((u0 + u1) / 2.0) + v * ((v0 + v1) / 2.0);
        let shape = Shape { center, angle: th, kind: Kind::Rounded { a: (u1 - u0) / 2.0, b: (v1 - v0) / 2.0, r: 0.0 } };
        if best.as_ref().is_none_or(|(a, _)| area < *a - 1e-12) {
            best = Some((area, shape));
        }
        if angle.is_some() {
            break;
        }
    }
    best.map_or(Shape { center: Vec2::ZERO, angle: 0.0, kind: Kind::Rounded { a: 0.0, b: 0.0, r: 0.0 } }, |(_, s)| s)
}

fn circle_through(a: Vec2, b: Vec2, c: Vec2) -> Option<(Vec2, f64)> {
    let (bx, by) = (b.x - a.x, b.y - a.y);
    let (cx, cy) = (c.x - a.x, c.y - a.y);
    let d = 2.0 * (bx * cy - by * cx);
    if d.abs() < 1e-12 {
        return None;
    }
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let ux = (cy * b2 - by * c2) / d;
    let uy = (bx * c2 - cx * b2) / d;
    let center = Vec2::new(a.x + ux, a.y + uy);
    Some((center, center.dist(a)))
}

/// The smallest circle around the points (Welzl's move-to-front, on the hull).
fn min_circle(h: &[Vec2]) -> Shape {
    let inside = |c: (Vec2, f64), p: Vec2| c.0.dist(p) <= c.1 + 1e-9;
    let two = |a: Vec2, b: Vec2| (a.lerp(b, 0.5), a.dist(b) / 2.0);
    let mut c = h.first().map_or((Vec2::ZERO, 0.0), |p| (*p, 0.0));
    for (i, &p) in h.iter().enumerate() {
        if inside(c, p) {
            continue;
        }
        c = (p, 0.0);
        for (j, &q) in h.iter().enumerate().take(i) {
            if inside(c, q) {
                continue;
            }
            c = two(p, q);
            for &r in h.iter().take(j) {
                if !inside(c, r) {
                    c = circle_through(p, q, r).unwrap_or_else(|| {
                        // Collinear: the farthest pair bounds all three.
                        let cands = [two(p, q), two(p, r), two(q, r)];
                        cands.into_iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap_or(c)
                    });
                }
            }
        }
    }
    Shape { center: c.0, angle: 0.0, kind: Kind::Circle { r: c.1 } }
}

/// The shortest straight distance between neighbouring points of a closed ring.
fn min_chord(pts: &[Vec2]) -> f64 {
    let n = pts.len();
    (0..n).filter_map(|i| pts.get(i).zip(pts.get((i + 1) % n))).map(|(a, b)| a.dist(*b)).fold(f64::INFINITY, f64::min)
}

/// Points evenly along the loop, `n` of them, the first `phase` (0 … 1) of a step in.
fn along(lp: &Loop2, n: usize, phase: f64) -> Vec<Vec2> {
    let lens: Vec<f64> = lp.segs.iter().map(Seg2::length).collect();
    let total: f64 = lens.iter().sum();
    if !(total > 1e-9) || n == 0 {
        return Vec::new();
    }
    let step = total / n as f64;
    let mut out = Vec::with_capacity(n);
    let (mut seg, mut start) = (0usize, 0.0);
    for i in 0..n {
        let target = (i as f64 + phase) * step;
        while seg + 1 < lp.segs.len() && start + lens.get(seg).copied().unwrap_or(0.0) < target {
            start += lens.get(seg).copied().unwrap_or(0.0);
            seg += 1;
        }
        let (Some(s), Some(&l)) = (lp.segs.get(seg), lens.get(seg)) else { break };
        let t = if l > 1e-12 { ((target - start) / l).clamp(0.0, 1.0) } else { 0.0 };
        out.push(s.point_at(t));
    }
    out
}

/// What a doubler is designed from.
struct Plan {
    thickness: f64,
    rivet_diameter: f64,
    edge_distance: f64,
    pitch: f64,
    row_pitch: f64,
    rows: usize,
    stagger: bool,
    circle: bool,
    margin: f64,
    corner_radius: f64,
    angle: Option<f64>,
}

/// A designed doubler.
struct Doubler {
    plan: Plan,
    damage: Vec<Vec2>,
    cutout: Shape,
    rows: Vec<(f64, Vec<Vec2>, f64)>,
    plate: Shape,
}

fn mm(p: &Value, k: &str, cmd: &str, default: f64, lo: f64, hi: f64) -> Result<f64> {
    match p.get(k).filter(|v| !v.is_null()) {
        None => Ok(default),
        Some(v) => match v.as_f64().filter(|x| x.is_finite() && *x >= lo && *x <= hi) {
            Some(x) => Ok(x),
            None => Err(bad(cmd, format!("`{k}` must be a number from {lo} to {hi} mm"))),
        },
    }
}

/// The damage outline: `damage` as given, or traced from `photo`.
fn damage_of(p: &Value, cmd: &str) -> Result<(Vec<Vec2>, Option<Value>)> {
    if let Some(photo) = p.get("photo").filter(|v| !v.is_null()) {
        if !photo.is_object() {
            return Err(bad(cmd, "`photo` must be an object with repair.trace_photo's parameters"));
        }
        let traced = trace_value(photo, cmd)?;
        let chosen = traced.get("chosen").and_then(Value::as_u64).unwrap_or(0) as usize;
        let outline = traced
            .get("regions")
            .and_then(Value::as_array)
            .and_then(|r| r.get(chosen))
            .and_then(|r| r.get("outline"))
            .and_then(Value::as_array)
            .ok_or_else(|| bad(cmd, "the photo has no mark to trace (lower `threshold`, or `invert` for light marks)"))?;
        let pts: Vec<Vec2> = outline.iter().filter_map(vec2).collect();
        return Ok((pts, Some(traced)));
    }
    let arr = p.get("damage").and_then(Value::as_array).ok_or_else(|| bad(cmd, "give `damage` ([[x,y], …] mm) or `photo`"))?;
    if arr.len() > 20_000 {
        return Err(bad(cmd, "`damage` has too many points (20000 at most)"));
    }
    let pts: Vec<Vec2> = arr
        .iter()
        .map(|v| vec2(v).filter(|q| q.is_finite() && q.x.abs() < 1e6 && q.y.abs() < 1e6))
        .collect::<Option<_>>()
        .ok_or_else(|| bad(cmd, "`damage` must be [[x,y], …] in mm"))?;
    Ok((pts, None))
}

fn plan_of(p: &Value, cmd: &str) -> Result<Plan> {
    let thickness = mm(p, "thickness", cmd, 1.0, 0.1, 50.0)?;
    let rivet_diameter = mm(p, "rivet_diameter", cmd, 3.2, 0.5, 50.0)?;
    let edge_distance = mm(p, "edge_distance", cmd, 2.5 * rivet_diameter, 0.5, 1000.0)?;
    let pitch = mm(p, "pitch", cmd, 5.0 * rivet_diameter, 0.5, 1000.0)?;
    let row_pitch = mm(p, "row_pitch", cmd, 0.75 * pitch, 0.5, 1000.0)?;
    if edge_distance < rivet_diameter {
        return Err(bad(cmd, "`edge_distance` must be at least the rivet diameter (2 × is the usual minimum)"));
    }
    if pitch < 2.0 * rivet_diameter || row_pitch < 2.0 * rivet_diameter {
        return Err(bad(cmd, "`pitch` and `row_pitch` must be at least twice the rivet diameter (3 × is the usual minimum)"));
    }
    let rows = match p.get("rows").filter(|v| !v.is_null()) {
        None => 2,
        Some(v) => match v.as_u64() {
            Some(n) if (1..=10).contains(&n) => n as usize,
            _ => return Err(bad(cmd, "`rows` must be 1 to 10")),
        },
    };
    let circle = match str_(p, "cutout").map(str::to_ascii_lowercase).as_deref() {
        None | Some("rectangle") | Some("rounded_rectangle") => false,
        Some("circle") => true,
        Some(_) => return Err(bad(cmd, "`cutout` must be rectangle or circle")),
    };
    let angle = match p.get("angle").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(
            v.as_f64().filter(|a| a.is_finite() && a.abs() <= 3600.0).ok_or_else(|| bad(cmd, "`angle` must be a number (degrees)"))?.to_radians(),
        ),
    };
    Ok(Plan {
        thickness,
        rivet_diameter,
        edge_distance,
        pitch,
        row_pitch,
        rows,
        stagger: bool_(p, "stagger").unwrap_or(true),
        circle,
        margin: mm(p, "cleanup_margin", cmd, 3.0, 0.0, 1000.0)?,
        corner_radius: mm(p, "corner_radius", cmd, 6.0, 0.0, 1000.0)?,
        angle,
    })
}

fn design(plan: Plan, damage: Vec<Vec2>, cmd: &str) -> Result<Doubler> {
    if damage.len() < 3 {
        return Err(bad(cmd, "the damage outline needs at least 3 points"));
    }
    let h = hull(&damage);
    if h.len() < 3 || Loop2::polygon(&h).signed_area().abs() < 1e-6 {
        return Err(bad(cmd, "the damage outline has no area"));
    }
    let bare = if plan.circle { min_circle(&h) } else { min_rect(&h, plan.angle) };
    let mut cutout = bare.offset(plan.margin);
    if let Kind::Rounded { a, b, r } = &mut cutout.kind {
        *r = plan.corner_radius.min(*a).min(*b);
    }
    // Rounding the corners cuts inside the rectangle: grow it until the whole damage is inside.
    cutout.grow_to_contain(&h);
    let (w, h) = cutout.size();
    if w > 1e5 || h > 1e5 {
        return Err(bad(cmd, "the damage is too large (100 m)"));
    }
    let mut rows = Vec::new();
    let mut total = 0usize;
    for i in 0..plan.rows {
        let d = plan.edge_distance + i as f64 * plan.row_pitch;
        let line = cutout.offset(d).outline();
        let len: f64 = line.segs.iter().map(Seg2::length).sum();
        let mut n = ((len / plan.pitch).ceil() as usize).max(3);
        total = total.saturating_add(n);
        if total > 5000 {
            return Err(bad(cmd, "more than 5000 rivets: raise `pitch` or lower `rows`"));
        }
        let phase = if plan.stagger && i % 2 == 1 { 0.5 } else { 0.0 };
        // Along a curve the holes sit closer than the developed spacing: fewer rivets until no
        // two are nearer than twice the diameter, and never further apart than the pitch.
        let mut pts = along(&line, n, phase);
        while n > 3 && min_chord(&pts) < 2.0 * plan.rivet_diameter - 1e-9 {
            n -= 1;
            pts = along(&line, n, phase);
        }
        if len / n as f64 > plan.pitch + 1e-9 {
            return Err(bad(
                cmd,
                format!(
                    "row {}: rivets `pitch` apart would have holes closer than twice the diameter around the corners; raise `pitch` or `corner_radius`",
                    i + 1
                ),
            ));
        }
        rows.push((d, pts, len / n as f64));
    }
    let last = plan.edge_distance + (plan.rows.saturating_sub(1)) as f64 * plan.row_pitch;
    let plate = cutout.offset(last + plan.edge_distance);
    Ok(Doubler { plan, damage, cutout, rows, plate })
}

impl Doubler {
    fn rivets(&self) -> Vec<Vec2> {
        self.rows.iter().flat_map(|(_, r, _)| r.iter().copied()).collect()
    }

    fn note(&self) -> String {
        let (w, h) = self.plate.size();
        format!(
            "DOUBLER {w:.1} x {h:.1} x {t:.2} MM - {n} RIVETS DIA {d:.2} - ED {ed:.1} PITCH {p:.1} ROWS {r} - CUT OUT DASHED",
            t = self.plan.thickness,
            n = self.rivets().len(),
            d = self.plan.rivet_diameter,
            ed = self.plan.edge_distance,
            p = self.plan.pitch,
            r = self.plan.rows
        )
    }

    fn dxf(&self, notes: bool) -> String {
        let layers = vec![
            DxfLayer::new("DOUBLER", 7),
            DxfLayer::new("RIVET", 1),
            DxfLayer::new("CUTOUT", 5).dashed(),
            DxfLayer::new("DAMAGE", 3).dashed(),
            DxfLayer::new("NOTES", 7),
        ];
        let mut ents = Vec::new();
        let r = self.plan.rivet_diameter / 2.0;
        for c in self.rivets() {
            ents.push(DxfEntity { layer: "RIVET".into(), shape: DxfShape::Circle { c, r } });
        }
        ents.extend(self.plate.dxf("DOUBLER"));
        ents.extend(self.cutout.dxf("CUTOUT"));
        let mut poly = self.damage.clone();
        if let Some(&first) = poly.first() {
            poly.push(first);
        }
        ents.push(DxfEntity { layer: "DAMAGE".into(), shape: DxfShape::Polyline { points: poly } });
        if notes {
            let (_, h) = self.plate.size();
            let at = self.plate.center - Vec2::new(0.0, h / 2.0 + 8.0);
            ents.push(DxfEntity { layer: "NOTES".into(), shape: DxfShape::Text { at, height: 3.0, angle_deg: 0.0, text: self.note() } });
        }
        solidcraft_io::write_dxf_drawing(&layers, &ents)
    }

    fn json(&self) -> Value {
        let (w, h) = self.plate.size();
        json!({
            "cutout": self.cutout.json(),
            "doubler": self.plate.json(),
            "size": [w, h],
            "thickness": self.plan.thickness,
            "rivets": self.rivets().iter().map(|p| pt(*p)).collect::<Vec<_>>(),
            "rivet_diameter": self.plan.rivet_diameter,
            "rows": self.rows.iter().map(|(d, r, p)| json!({"offset": d, "count": r.len(), "pitch": p})).collect::<Vec<_>>(),
            "damage": {"points": self.damage.len(), "area": Loop2::polygon(&self.damage).signed_area().abs()},
            "note": self.note(),
        })
    }
}

fn design_of(p: &Value, cmd: &str) -> Result<(Doubler, Option<Value>)> {
    let plan = plan_of(p, cmd)?;
    let (damage, traced) = damage_of(p, cmd)?;
    Ok((design(plan, damage, cmd)?, traced))
}

fn doubler(_: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "repair.doubler";
    let (d, traced) = design_of(p, cmd)?;
    let notes = bool_(p, "notes").unwrap_or(true);
    let text = d.dxf(notes);
    let mut out = d.json();
    if let Some(t) = traced {
        out["photo"] = json!({"width_px": t.get("width_px"), "height_px": t.get("height_px"), "mm_per_pixel": t.get("mm_per_pixel"), "chosen": t.get("chosen"), "regions": t.get("regions").and_then(Value::as_array).map_or(0, Vec::len)});
    }
    match str_(p, "path").filter(|x| !x.trim().is_empty()) {
        Some(path) => {
            solidcraft_io::vfs::write(path, text.as_bytes()).map_err(|e| EngineError::Other(format!("{path}: {e}")))?;
            out["path"] = json!(path);
        }
        None => out["dxf"] = json!(text),
    }
    Ok(out)
}

fn doubler_sketch(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "repair.doubler_sketch";
    let (d, _) = design_of(p, cmd)?;
    let off = p.get("at").and_then(vec2).filter(|q| q.is_finite()).unwrap_or(Vec2::ZERO);
    let mut created = None;
    if s.active_sketch.is_none() {
        let mut cp =
            json!({"plane": p.get("plane").cloned().unwrap_or(json!("XY")), "project_edges": false, "name": str_(p, "name").unwrap_or("Doubler")});
        if let Some(n) = str_(p, "name") {
            cp["name"] = json!(n);
        }
        let r = (super::find_command("sketch.create").ok_or_else(|| bad(cmd, "sketch.create"))?.run)(s, &cp)?;
        created = r.get("sketch").and_then(Value::as_u64);
    }
    let outline = d.plate.outline();
    let rivets = d.rivets();
    let r = d.plan.rivet_diameter / 2.0;
    let (n_curves, info) = edit(s, p, cmd, false, |sk, _| {
        let mut n = 0usize;
        // The outline's corners are shared points, so it is one closed profile.
        let m = outline.segs.len();
        let mut starts: Vec<usize> = Vec::with_capacity(m);
        for seg in &outline.segs {
            starts.push(sk.add_point(seg.start() + off, None)?);
        }
        for (i, seg) in outline.segs.iter().enumerate() {
            let (Some(&a), Some(&b)) = (starts.get(i), starts.get((i + 1) % m)) else { continue };
            match *seg {
                Seg2::Line { a: p0, b: p1 } => {
                    sk.add_line(p0 + off, p1 + off, Some(a), Some(b), None)?;
                }
                Seg2::Arc { center, .. } => {
                    sk.add_arc(center + off, seg.start() + off, seg.end() + off, [None, Some(a), Some(b)], None)?;
                }
                _ => continue,
            }
            n += 1;
        }
        for c in &rivets {
            sk.add_circle(*c + off, r, None, None)?;
            n += 1;
        }
        Ok(n)
    })?;
    let id = s.active_sketch.ok_or_else(|| bad(cmd, "no sketch"))?;
    let (w, h) = d.plate.size();
    // A point on the plate (between the trim line and the first rivet row) for `profiles`.
    let on_plate = d.cutout.offset(d.plan.edge_distance / 2.0).outline().segs.first().map_or(d.cutout.center, Seg2::start) + off;
    Ok(
        json!({"sketch": id, "created_sketch": created, "curves": n_curves, "rivets": rivets.len(), "size": [w, h], "profile_point": pt(on_plate), "info": info}),
    )
}

fn photo_bytes(p: &Value, cmd: &str) -> Result<Vec<u8>> {
    let b = if let Some(d) = str_(p, "data") {
        if d.len() > MAX_PHOTO_BYTES * 4 / 3 + 16 {
            return Err(bad(cmd, "the photo is too large"));
        }
        base64_decode(d).ok_or_else(|| bad(cmd, "`data` is not base64"))?
    } else {
        let path = str_(p, "path").filter(|x| !x.trim().is_empty() && x.len() < 4096).ok_or_else(|| bad(cmd, "give `path` or `data`"))?;
        solidcraft_io::vfs::read(path, MAX_PHOTO_BYTES).map_err(|e| EngineError::Other(format!("{path}: {e}")))?
    };
    if b.is_empty() || b.len() > MAX_PHOTO_BYTES {
        return Err(bad(cmd, "the photo must be 1 byte to 16 MiB"));
    }
    Ok(b)
}

/// `repair.trace_photo` on `p`: the marks in millimetres.
fn trace_value(p: &Value, cmd: &str) -> Result<Value> {
    let bytes = photo_bytes(p, cmd)?;
    let threshold = mm(p, "threshold", cmd, 0.5, 0.0, 1.0)?;
    let invert = bool_(p, "invert").unwrap_or(false);
    let despeckle = bool_(p, "despeckle").unwrap_or(true);
    let min_area = mm(p, "min_area", cmd, 1.0, 0.0, 1e9)?;
    let simplify = mm(p, "simplify", cmd, 0.5, 0.0, 1000.0)?;
    let photo = photo_trace::decode_luma(&bytes).map_err(|e| bad(cmd, e.to_string()))?;
    // Every pixel coordinate given or returned is in the photo's own pixels; a large photo is
    // traced on a grid `f` times coarser.
    let f = f64::from(photo.downsampled_by);
    let (w, h, luma) = (photo.width, photo.height, &photo.luma);
    let [ow, oh] = photo.original;
    // The scale: mm per photo pixel.
    let k_photo = if let Some(k) = p.get("mm_per_pixel").filter(|v| !v.is_null()) {
        k.as_f64().filter(|k| k.is_finite() && *k > 1e-6 && *k < 1e4).ok_or_else(|| bad(cmd, "`mm_per_pixel` must be a positive number"))?
    } else if let Some(wm) = p.get("width_mm").filter(|v| !v.is_null()) {
        let wm = wm.as_f64().filter(|w| w.is_finite() && *w > 1e-3 && *w < 1e7).ok_or_else(|| bad(cmd, "`width_mm` must be a positive number"))?;
        wm / f64::from(ow)
    } else if let Some(c) = p.get("calibrate").filter(|v| !v.is_null()) {
        let a = c.get("a").and_then(vec2).filter(|q| q.is_finite()).ok_or_else(|| bad(cmd, "`calibrate.a` must be [px, py]"))?;
        let b = c.get("b").and_then(vec2).filter(|q| q.is_finite()).ok_or_else(|| bad(cmd, "`calibrate.b` must be [px, py]"))?;
        let dist =
            num(c, "distance").filter(|d| *d > 1e-3 && *d < 1e7).ok_or_else(|| bad(cmd, "`calibrate.distance` must be a positive length (mm)"))?;
        let px = a.dist(b);
        if px < 1.0 {
            return Err(bad(cmd, "`calibrate.a` and `calibrate.b` must be different pixels"));
        }
        dist / px
    } else {
        return Err(bad(cmd, "give the scale: `mm_per_pixel`, `width_mm` or `calibrate: {a, b, distance}`"));
    };
    // mm per grid pixel.
    let k = k_photo * f;
    let o = TraceOptions {
        threshold,
        invert,
        despeckle,
        min_area_px: ((min_area / (k * k)).ceil().clamp(1.0, 1e9)) as usize,
        max_area_px: usize::MAX,
        simplify_px: simplify / k,
    };
    let t = photo_trace::trace_luma(w, h, luma, &o).map_err(|e| bad(cmd, e.to_string()))?;
    // Which mark: by index, by a pixel on it, or the largest.
    let chosen = if let Some(r) = p.get("region").filter(|v| !v.is_null()) {
        let i = r.as_u64().ok_or_else(|| bad(cmd, "`region` must be an index"))? as usize;
        if i >= t.regions.len() {
            return Err(bad(cmd, format!("`region` {i}: the photo has {} mark(s)", t.regions.len())));
        }
        i
    } else if let Some(at) = p.get("at").filter(|v| !v.is_null()) {
        let q = vec2(at).filter(|q| q.is_finite()).ok_or_else(|| bad(cmd, "`at` must be [px, py]"))? * (1.0 / f);
        t.regions
            .iter()
            .position(|r| r.bbox[0] <= q.x && q.x <= r.bbox[2] && r.bbox[1] <= q.y && q.y <= r.bbox[3] && Loop2::polygon(&r.outline).contains(q))
            .ok_or_else(|| bad(cmd, "no mark at `at`"))?
    } else {
        0
    };
    // The origin, in pixels (y down).
    let origin = match p.get("origin").filter(|v| !v.is_null()) {
        None => None,
        Some(Value::String(s)) if s.eq_ignore_ascii_case("center") => None,
        Some(Value::String(s)) if s.eq_ignore_ascii_case("image") => Some(Vec2::new(0.0, f64::from(h))),
        Some(v) => Some(vec2(v).filter(|q| q.is_finite()).ok_or_else(|| bad(cmd, "`origin` must be center, image or [px, py]"))? * (1.0 / f)),
    };
    let origin = origin.unwrap_or_else(|| {
        t.regions
            .get(chosen)
            .map_or(Vec2::new(f64::from(w) / 2.0, f64::from(h) / 2.0), |r| Vec2::new((r.bbox[0] + r.bbox[2]) / 2.0, (r.bbox[1] + r.bbox[3]) / 2.0))
    });
    let to_mm = |q: Vec2| Vec2::new((q.x - origin.x) * k, (origin.y - q.y) * k);
    let regions: Vec<Value> = t
        .regions
        .iter()
        .map(|r| {
            let lo = to_mm(Vec2::new(r.bbox[0], r.bbox[3]));
            let hi = to_mm(Vec2::new(r.bbox[2], r.bbox[1]));
            json!({
                "area_mm2": r.area_px as f64 * k * k,
                "area_px": r.area_px,
                "bbox": [lo.x, lo.y, hi.x, hi.y],
                "bbox_px": [r.bbox[0] * f, r.bbox[1] * f, r.bbox[2] * f, r.bbox[3] * f],
                "center_px": [(r.bbox[0] + r.bbox[2]) / 2.0 * f, (r.bbox[1] + r.bbox[3]) / 2.0 * f],
                "outline": r.outline.iter().map(|q| pt(to_mm(*q))).collect::<Vec<_>>(),
            })
        })
        .collect();
    Ok(json!({
        "width_px": ow, "height_px": oh, "mm_per_pixel": k_photo, "downsampled_by": photo.downsampled_by,
        "width_mm": f64::from(ow) * k_photo, "height_mm": f64::from(oh) * k_photo,
        "chosen": chosen, "regions": regions,
    }))
}

/// A list of `[px, py]` photo pixels (3 or more), finite.
fn pixel_polygon(v: &Value, cmd: &str, k: &str) -> Result<Vec<Vec2>> {
    let arr = v.as_array().ok_or_else(|| bad(cmd, format!("`{k}` must be [[px,py], …]")))?;
    if arr.len() < 3 || arr.len() > 10_000 {
        return Err(bad(cmd, format!("`{k}` needs 3 to 10000 points")));
    }
    arr.iter()
        .map(|q| vec2(q).filter(|q| q.is_finite() && q.x.abs() < 1e7 && q.y.abs() < 1e7))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| bad(cmd, format!("`{k}` must be [[px,py], …]")))
}

fn skin_from_photo(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "repair.skin_from_photo";
    let bytes = photo_bytes(p, cmd)?;
    let photo = photo_trace::decode_luma(&bytes).map_err(|e| bad(cmd, e.to_string()))?;
    let f = f64::from(photo.downsampled_by);
    let [ow, oh] = photo.original;
    // Photo pixel → mm with y down: a plain scale, or a homography through a known rectangle.
    let (to_mm_ydown, k_photo, default_outline): (Box<dyn Fn(Vec2) -> Option<Vec2>>, f64, Vec<Vec2>) = if let Some(r) =
        p.get("rectify").filter(|v| !v.is_null())
    {
        let corners = r.get("corners").map(|c| pixel_polygon(c, cmd, "rectify.corners")).transpose()?.unwrap_or_default();
        let [c0, c1, c2, c3] = <[Vec2; 4]>::try_from(corners.clone()).map_err(|_| bad(cmd, "`rectify.corners` must be four [px,py] corners"))?;
        let wm = num(r, "width_mm").filter(|w| *w > 1e-3 && *w < 1e7).ok_or_else(|| bad(cmd, "`rectify.width_mm` must be a positive length (mm)"))?;
        let hm =
            num(r, "height_mm").filter(|h| *h > 1e-3 && *h < 1e7).ok_or_else(|| bad(cmd, "`rectify.height_mm` must be a positive length (mm)"))?;
        let h = photo_trace::homography([c0, c1, c2, c3], [Vec2::ZERO, Vec2::new(wm, 0.0), Vec2::new(wm, hm), Vec2::new(0.0, hm)])
            .ok_or_else(|| bad(cmd, "`rectify.corners` must be the four corners of a rectangle (no three in line)"))?;
        let area_px = Loop2::polygon(&corners).signed_area().abs();
        if area_px < 4.0 {
            return Err(bad(cmd, "`rectify.corners` enclose no area"));
        }
        let k = (wm * hm / area_px).sqrt();
        (Box::new(move |q| photo_trace::map_point(&h, q)), k, corners)
    } else {
        let k = if let Some(k) = p.get("mm_per_pixel").filter(|v| !v.is_null()) {
            k.as_f64().filter(|k| k.is_finite() && *k > 1e-6 && *k < 1e4).ok_or_else(|| bad(cmd, "`mm_per_pixel` must be a positive number"))?
        } else if let Some(wm) = p.get("width_mm").filter(|v| !v.is_null()) {
            let wm =
                wm.as_f64().filter(|w| w.is_finite() && *w > 1e-3 && *w < 1e7).ok_or_else(|| bad(cmd, "`width_mm` must be a positive number"))?;
            wm / f64::from(ow)
        } else if let Some(c) = p.get("calibrate").filter(|v| !v.is_null()) {
            let a = c.get("a").and_then(vec2).filter(|q| q.is_finite()).ok_or_else(|| bad(cmd, "`calibrate.a` must be [px, py]"))?;
            let b = c.get("b").and_then(vec2).filter(|q| q.is_finite()).ok_or_else(|| bad(cmd, "`calibrate.b` must be [px, py]"))?;
            let dist = num(c, "distance")
                .filter(|d| *d > 1e-3 && *d < 1e7)
                .ok_or_else(|| bad(cmd, "`calibrate.distance` must be a positive length (mm)"))?;
            if a.dist(b) < 1.0 {
                return Err(bad(cmd, "`calibrate.a` and `calibrate.b` must be different pixels"));
            }
            dist / a.dist(b)
        } else {
            return Err(bad(cmd, "give the scale: `calibrate: {a, b, distance}`, `mm_per_pixel`, `width_mm` or `rectify`"));
        };
        let whole = vec![Vec2::ZERO, Vec2::new(f64::from(ow), 0.0), Vec2::new(f64::from(ow), f64::from(oh)), Vec2::new(0.0, f64::from(oh))];
        (Box::new(move |q| Some(q * k)), k, whole)
    };
    let outline_px = match p.get("outline").filter(|v| !v.is_null()) {
        Some(v) => pixel_polygon(v, cmd, "outline")?,
        None => default_outline,
    };
    let outline_poly = Loop2::polygon(&outline_px);
    if outline_poly.signed_area().abs() < 4.0 {
        return Err(bad(cmd, "`outline` encloses no area"));
    }
    // The outline in mm (y down for now).
    let outline_dn: Vec<Vec2> =
        outline_px.iter().map(|q| to_mm_ydown(*q)).collect::<Option<_>>().ok_or_else(|| bad(cmd, "`outline` reaches the photo's horizon"))?;
    let (x0, y1) = outline_dn.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(x0, y1), q| (x0.min(q.x), y1.max(q.y)));
    let to_mm = |q: Vec2| to_mm_ydown(q).map(|m| Vec2::new(m.x - x0, y1 - m.y));
    let outline_mm: Vec<Vec2> = outline_dn.iter().map(|m| Vec2::new(m.x - x0, y1 - m.y)).collect();
    let (w, h) = outline_mm.iter().fold((0.0f64, 0.0f64), |(w, h), q| (w.max(q.x), h.max(q.y)));
    if w > 1e5 || h > 1e5 {
        return Err(bad(cmd, "the region is too large (100 m)"));
    }
    // The holes.
    let want_holes = bool_(p, "holes").unwrap_or(true);
    let expected = mm(p, "hole_diameter", cmd, 3.2, 0.2, 500.0)?;
    let drawn = match p.get("drawn_diameter").filter(|v| !v.is_null()) {
        None => None,
        Some(_) => Some(mm(p, "drawn_diameter", cmd, expected, 0.1, 500.0)?),
    };
    let (dark, light) = match str_(p, "polarity").map(str::to_ascii_lowercase).as_deref() {
        None | Some("both") => (true, true),
        Some("dark") => (true, false),
        Some("light") => (false, true),
        Some(_) => return Err(bad(cmd, "`polarity` must be dark, light or both")),
    };
    let threshold = match p.get("threshold").filter(|v| !v.is_null()) {
        None => None,
        Some(_) => Some(mm(p, "threshold", cmd, 0.5, 0.0, 1.0)?),
    };
    let mut holes: Vec<(Vec2, f64, Vec2, bool)> = Vec::new();
    if want_holes {
        let k_grid = k_photo * f;
        let o = photo_trace::BlobOptions { threshold, diameter_px: [expected * 0.5 / k_grid, expected * 2.0 / k_grid], dark, light };
        let found = photo_trace::blobs(photo.width, photo.height, &photo.luma, &o).map_err(|e| bad(cmd, e.to_string()))?;
        for b in found {
            let c_px = b.center * f;
            if !outline_poly.contains(c_px) {
                continue;
            }
            let Some(c_mm) = to_mm(c_px) else { continue };
            let d = drawn.unwrap_or(b.diameter_px * k_grid);
            // Inside the outline by at least its radius, so the sketch stays one profile.
            if c_mm.x < d / 2.0 || c_mm.y < d / 2.0 || c_mm.x > w - d / 2.0 || c_mm.y > h - d / 2.0 {
                continue;
            }
            if holes.len() >= 2000 {
                return Err(bad(cmd, "more than 2000 holes: raise `hole_diameter` or narrow `outline`"));
            }
            holes.push((c_mm, d, c_px, b.dark));
        }
    }
    // The sheet rule.
    if let Some(t) = p.get("thickness").filter(|v| !v.is_null()) {
        let t = t.as_f64().filter(|t| t.is_finite() && *t > 0.01 && *t < 100.0).ok_or_else(|| bad(cmd, "`thickness` must be a length in mm"))?;
        let rule = super::find_command("sheet.manage_rules").ok_or_else(|| bad(cmd, "sheet.manage_rules"))?;
        rule.run_nested(s, &json!({"name": "Skin", "thickness": format!("{t} mm"), "active": true}))?;
    }
    // The sketch: the outline and a circle per hole.
    let name = str_(p, "name").unwrap_or("Skin");
    let cp = json!({"plane": p.get("plane").cloned().unwrap_or(json!("XY")), "project_edges": false, "name": name});
    let r = (super::find_command("sketch.create").ok_or_else(|| bad(cmd, "sketch.create"))?.run)(s, &cp)?;
    let sketch = r.get("sketch").and_then(Value::as_u64).ok_or_else(|| bad(cmd, "sketch.create gave no sketch"))?;
    let outline_sk = outline_mm.clone();
    let holes_sk: Vec<(Vec2, f64)> = holes.iter().map(|(c, d, _, _)| (*c, *d)).collect();
    let (n_curves, _) = edit(s, p, cmd, false, |sk, _| {
        let m = outline_sk.len();
        let mut ids = Vec::with_capacity(m);
        for q in &outline_sk {
            ids.push(sk.add_point(*q, None)?);
        }
        let mut n = 0;
        for i in 0..m {
            let (Some(&a), Some(&b)) = (ids.get(i), ids.get((i + 1) % m)) else { continue };
            let (Some(&pa), Some(&pb)) = (outline_sk.get(i), outline_sk.get((i + 1) % m)) else { continue };
            if pa.dist(pb) < 1e-9 {
                continue;
            }
            sk.add_line(pa, pb, Some(a), Some(b), None)?;
            n += 1;
        }
        for (c, d) in &holes_sk {
            sk.add_circle(*c, d / 2.0, None, None)?;
            n += 1;
        }
        Ok(n)
    })?;
    (super::find_command("sketch.finish").ok_or_else(|| bad(cmd, "sketch.finish"))?.run)(s, &json!({}))?;
    // The sheet body, from a point on the skin that is in no hole.
    let mut body = Value::Null;
    if bool_(p, "body").unwrap_or(true) {
        let centroid = Loop2::polygon(&outline_mm).centroid();
        let clear =
            |q: Vec2| outline_mm.len() >= 3 && Loop2::polygon(&outline_mm).contains(q) && holes.iter().all(|(c, d, _, _)| c.dist(q) > d / 2.0 + 0.05);
        let mut pick = clear(centroid).then_some(centroid);
        let mut step = 1.0;
        while pick.is_none() && step < w.max(h) {
            for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)] {
                let q = centroid + Vec2::new(dx * step, dy * step);
                if clear(q) {
                    pick = Some(q);
                    break;
                }
            }
            step *= 1.5;
        }
        let point = pick.ok_or_else(|| bad(cmd, "no point of the skin is clear of its holes"))?;
        let fl = super::find_command("sheet.flange").ok_or_else(|| bad(cmd, "sheet.flange"))?;
        let fr = fl.run_nested(s, &json!({"sketch": sketch, "profiles": [{"point": [point.x, point.y]}], "body_name": name}))?;
        body = fr.get("bodies").and_then(|b| b.get(0)).cloned().unwrap_or(Value::Null);
    }
    Ok(json!({
        "sketch": sketch, "body": body, "curves": n_curves,
        "outline": outline_mm.iter().map(|q| pt(*q)).collect::<Vec<_>>(), "size": [w, h],
        "holes": holes.iter().map(|(c, d, px, dark)| json!({"center": pt(*c), "diameter": d, "center_px": pt(*px), "dark": dark})).collect::<Vec<_>>(),
        "mm_per_pixel": k_photo, "width_px": ow, "height_px": oh,
    }))
}

fn trace_photo(_: &mut Session, p: &Value) -> Result<Value> {
    trace_value(p, "repair.trace_photo")
}

#[cfg(test)]
#[path = "repair_tests.rs"]
mod tests;

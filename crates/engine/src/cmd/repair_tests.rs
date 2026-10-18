//! Repair doublers: the clean-out follows the damage, the rivet rows keep their edge distances,
//! the DXF carries the plate, its holes and the trim line, a photo of the marked-up skin traces
//! to the same input, and the sketch form carries on into sheet.flange and the flat pattern.

use serde_json::{Value, json};
use solidcraft_geom::{Loop2, Vec2};
use solidcraft_io::photo_trace::gray_png;

use crate::{EngineError, Session};

fn run(s: &mut Session, id: &str, p: Value) -> Value {
    match s.execute(id, &p) {
        Ok(v) => v,
        Err(e) => panic!("{id} {p}: {e}"),
    }
}

fn fails(s: &mut Session, id: &str, p: Value) -> String {
    match s.execute(id, &p) {
        Ok(v) => panic!("{id} {p} succeeded: {v}"),
        Err(EngineError::Internal(_, m)) => panic!("{id} {p} panicked: {m}"),
        Err(e) => e.to_string(),
    }
}

fn pts(v: &Value) -> Vec<Vec2> {
    v.as_array().unwrap_or(&Vec::new()).iter().filter_map(crate::params::vec2).collect()
}

/// The segments of a shape as a loop.
fn shape_loop(v: &Value) -> Loop2 {
    let segs = v["segments"].as_array().cloned().unwrap_or_default();
    let segs = segs
        .iter()
        .filter_map(|s| {
            if let Some(l) = s.get("line") {
                let p = pts(l);
                return Some(solidcraft_geom::Seg2::Line { a: *p.first()?, b: *p.get(1)? });
            }
            let a = s.get("arc")?;
            Some(solidcraft_geom::Seg2::Arc {
                center: crate::params::vec2(&a["center"])?,
                radius: a["radius"].as_f64()?,
                start: a["start_deg"].as_f64()?.to_radians(),
                sweep: a["sweep_deg"].as_f64()?.to_radians(),
            })
        })
        .collect();
    Loop2 { segs }
}

fn dist_to_loop(lp: &Loop2, p: Vec2) -> f64 {
    lp.segs.iter().map(|s| s.point_at(s.closest_param(p)).dist(p)).fold(f64::INFINITY, f64::min)
}

/// A DXF's entity types with their layers.
fn dxf_entities(dxf: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = dxf.lines().collect();
    let pairs: Vec<(i32, &str)> = lines.chunks(2).filter_map(|c| Some((c.first()?.trim().parse().ok()?, c.get(1)?.trim()))).collect();
    let start = pairs.iter().position(|p| *p == (2, "ENTITIES")).unwrap_or(0);
    let mut out: Vec<(String, String)> = Vec::new();
    for (code, value) in pairs.iter().skip(start + 1) {
        match (*code, *value) {
            (0, "ENDSEC") => break,
            (0, kind) => out.push((kind.to_string(), String::new())),
            (8, layer) => {
                if let Some(last) = out.last_mut() {
                    last.1 = layer.to_string();
                }
            }
            _ => {}
        }
    }
    out
}

#[test]
fn a_rectangular_damage_gets_a_rounded_doubler_with_two_staggered_rows() {
    let mut s = Session::default();
    let d = run(&mut s, "repair.doubler", json!({"damage": [[-20, -5], [20, -5], [20, 5], [-20, 5]], "thickness": 1.0, "rivet_diameter": 3.2}));
    // Clean-out: the damage box plus 3 mm all round, 6 mm corners.
    assert_eq!(d["cutout"]["type"], "rounded_rectangle");
    assert!((d["cutout"]["width"].as_f64().unwrap() - 46.0).abs() < 1e-9, "{}", d["cutout"]);
    assert!((d["cutout"]["height"].as_f64().unwrap() - 16.0).abs() < 1e-9);
    assert!((d["cutout"]["radius"].as_f64().unwrap() - 6.0).abs() < 1e-9);
    // Defaults: ED 8, pitch 16, row pitch 12, two rows: the plate is the cutout + 2 × (8 + 12 + 8).
    let (ed, pitch, rp) = (8.0, 16.0, 12.0);
    assert!((d["doubler"]["width"].as_f64().unwrap() - (46.0 + 2.0 * (ed + rp + ed))).abs() < 1e-9, "{}", d["doubler"]);
    assert!((d["doubler"]["height"].as_f64().unwrap() - (16.0 + 2.0 * (ed + rp + ed))).abs() < 1e-9);
    let rows = d["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert!((rows[0]["offset"].as_f64().unwrap() - ed).abs() < 1e-9 && (rows[1]["offset"].as_f64().unwrap() - (ed + rp)).abs() < 1e-9);
    for r in rows {
        assert!(r["pitch"].as_f64().unwrap() <= pitch + 1e-9 && r["pitch"].as_f64().unwrap() > 0.6 * pitch, "{r}");
    }
    let rivets = pts(&d["rivets"]);
    assert_eq!(rivets.len(), rows.iter().map(|r| r["count"].as_u64().unwrap() as usize).sum::<usize>());
    // Every rivet keeps its edge distance from the trim line and the plate's edge, and its pitch from the others.
    let cutout = shape_loop(&d["cutout"]);
    let plate = shape_loop(&d["doubler"]);
    for &r in &rivets {
        assert!(dist_to_loop(&cutout, r) >= ed - 1e-6, "rivet {r:?} too close to the cutout");
        assert!(dist_to_loop(&plate, r) >= ed - 1e-6, "rivet {r:?} too close to the edge");
        assert!(plate.contains(r) && !cutout.contains(r));
    }
    for (i, a) in rivets.iter().enumerate() {
        for b in rivets.iter().skip(i + 1) {
            assert!(a.dist(*b) >= 2.0 * 3.2, "rivets {a:?} and {b:?} overlap");
        }
    }
    // Staggered: the second row's first rivet sits half a step along.
    let first_row = pts(&json!(rivets.iter().take(rows[0]["count"].as_u64().unwrap() as usize).map(|p| json!([p.x, p.y])).collect::<Vec<_>>()));
    let second = rivets.get(first_row.len()).copied().unwrap();
    let nearest = first_row.iter().map(|p| p.dist(second)).fold(f64::INFINITY, f64::min);
    assert!(nearest > rp && nearest < rp + pitch, "{nearest}");
    // The DXF: a circle per rivet on RIVET, the plate on DOUBLER, the trim line on CUTOUT, the damage on DAMAGE, a note.
    let ents = dxf_entities(d["dxf"].as_str().unwrap());
    let on = |layer: &str, kind: &str| ents.iter().filter(|(k, l)| k == kind && l == layer).count();
    assert_eq!(on("RIVET", "CIRCLE"), rivets.len());
    assert_eq!((on("DOUBLER", "LINE"), on("DOUBLER", "ARC")), (4, 4));
    assert_eq!((on("CUTOUT", "LINE"), on("CUTOUT", "ARC")), (4, 4));
    assert_eq!(on("DAMAGE", "LWPOLYLINE") + on("DAMAGE", "POLYLINE"), 1);
    assert_eq!(on("NOTES", "TEXT"), 1);
    assert!(d["note"].as_str().unwrap().contains("RIVETS DIA 3.20"));
    // Without notes.
    let quiet = run(&mut s, "repair.doubler", json!({"damage": [[-20, -5], [20, -5], [20, 5], [-20, 5]], "notes": false}));
    assert!(!dxf_entities(quiet["dxf"].as_str().unwrap()).iter().any(|(k, _)| k == "TEXT"));
}

#[test]
fn a_round_cleanout_and_a_tilted_damage() {
    let mut s = Session::default();
    // A 30 × 30 square damage: the smallest circle has the half diagonal as radius, plus the margin.
    let d =
        run(&mut s, "repair.doubler", json!({"damage": [[0, 0], [30, 0], [30, 30], [0, 30]], "cutout": "circle", "cleanup_margin": 2, "rows": 1}));
    assert_eq!(d["cutout"]["type"], "circle");
    let r = d["cutout"]["radius"].as_f64().unwrap();
    assert!((r - (15.0 * 2f64.sqrt() + 2.0)).abs() < 1e-6, "{r}");
    let c = pts(&json!([d["cutout"]["center"]]))[0];
    assert!(c.dist(Vec2::new(15.0, 15.0)) < 1e-6);
    assert!((d["doubler"]["radius"].as_f64().unwrap() - (r + 16.0)).abs() < 1e-6);
    let rivets = pts(&d["rivets"]);
    assert!(rivets.iter().all(|p| (p.dist(c) - (r + 8.0)).abs() < 1e-6));
    let ents = dxf_entities(d["dxf"].as_str().unwrap());
    assert_eq!(ents.iter().filter(|(k, l)| k == "CIRCLE" && l == "DOUBLER").count(), 1);
    // A 40 × 10 rectangle turned 30°: the tightest rectangle turns with it.
    let (sn, cs) = 30f64.to_radians().sin_cos();
    let rot = |x: f64, y: f64| json!([x * cs - y * sn, x * sn + y * cs]);
    let damage = json!([rot(-20.0, -5.0), rot(20.0, -5.0), rot(20.0, 5.0), rot(-20.0, 5.0), rot(0.0, 2.0)]);
    let d = run(&mut s, "repair.doubler", json!({"damage": damage, "cleanup_margin": 0, "corner_radius": 0}));
    let (w, h) = (d["cutout"]["width"].as_f64().unwrap(), d["cutout"]["height"].as_f64().unwrap());
    assert!((w.max(h) - 40.0).abs() < 1e-6 && (w.min(h) - 10.0).abs() < 1e-6, "{w} × {h}");
    let a = d["cutout"]["angle_deg"].as_f64().unwrap().rem_euclid(180.0);
    assert!((a - 30.0).abs() < 1e-6 || (a - 120.0).abs() < 1e-6, "{a}");
    // With the direction given, the box is axis-aligned and larger.
    let d = run(&mut s, "repair.doubler", json!({"damage": damage, "angle": 0, "cleanup_margin": 0, "corner_radius": 0}));
    assert!(d["cutout"]["width"].as_f64().unwrap() > 36.0 && d["cutout"]["height"].as_f64().unwrap() > 20.0, "{}", d["cutout"]);
    assert_eq!(d["cutout"]["angle_deg"], 0.0);
}

/// A white 200 × 150 photo with a black 40 × 20 mark at (60, 50) and a 6 × 6 dot at (150, 120).
fn photo() -> String {
    let (w, h) = (200u32, 150u32);
    let mut luma = vec![255u8; (w * h) as usize];
    for y in 50..70 {
        for x in 60..100 {
            luma[(y * w + x) as usize] = 0;
        }
    }
    for y in 120..126 {
        for x in 150..156 {
            luma[(y * w + x) as usize] = 0;
        }
    }
    solidcraft_doc::canvas::base64_encode(&gray_png(w, h, &luma).unwrap())
}

#[test]
fn a_photo_traces_to_millimetres_and_feeds_the_doubler() {
    let mut s = Session::default();
    let t = run(&mut s, "repair.trace_photo", json!({"data": photo(), "mm_per_pixel": 0.5}));
    assert_eq!((t["width_px"].as_u64(), t["height_px"].as_u64()), (Some(200), Some(150)));
    let regions = t["regions"].as_array().unwrap();
    assert_eq!(regions.len(), 2, "{t}");
    assert_eq!(t["chosen"], 0);
    // The big mark is 20 × 10 mm, centred on the origin, counter-clockwise with y up.
    let outline = pts(&regions[0]["outline"]);
    assert_eq!(outline.len(), 4, "{outline:?}");
    let lp = Loop2::polygon(&outline);
    assert!((lp.signed_area() - 200.0).abs() < 1e-9 && lp.is_ccw(), "{outline:?}");
    assert!(outline.iter().all(|p| (p.x.abs() - 10.0).abs() < 1e-9 && (p.y.abs() - 5.0).abs() < 1e-9));
    assert!((regions[0]["area_mm2"].as_f64().unwrap() - 200.0).abs() < 1e-9);
    assert!((regions[1]["area_mm2"].as_f64().unwrap() - 9.0).abs() < 1e-9);
    // The dot: by index, or by a pixel on it; the image origin puts the photo's bottom-left at 0,0.
    let t = run(&mut s, "repair.trace_photo", json!({"data": photo(), "mm_per_pixel": 0.5, "at": [152, 122], "origin": "image"}));
    assert_eq!(t["chosen"], 1);
    let dot = pts(&t["regions"][1]["outline"]);
    assert!(dot.iter().all(|p| p.x >= 75.0 - 1e-9 && p.x <= 78.0 + 1e-9 && p.y >= 12.0 - 1e-9 && p.y <= 15.0 + 1e-9), "{dot:?}");
    assert_eq!(run(&mut s, "repair.trace_photo", json!({"data": photo(), "width_mm": 100, "region": 1}))["chosen"], 1);
    // Calibrating on two points 40 px apart that are 10 mm apart gives 0.25 mm per pixel.
    let t = run(&mut s, "repair.trace_photo", json!({"data": photo(), "calibrate": {"a": [10, 10], "b": [50, 10], "distance": 10}}));
    assert!((t["mm_per_pixel"].as_f64().unwrap() - 0.25).abs() < 1e-12);
    assert!((t["regions"][0]["area_mm2"].as_f64().unwrap() - 50.0).abs() < 1e-9);
    // Too small a mark is noise.
    let t = run(&mut s, "repair.trace_photo", json!({"data": photo(), "mm_per_pixel": 0.5, "min_area": 20}));
    assert_eq!(t["regions"].as_array().unwrap().len(), 1);
    // The doubler straight from the photo.
    let d = run(&mut s, "repair.doubler", json!({"photo": {"data": photo(), "mm_per_pixel": 0.5}, "rows": 1}));
    assert!(
        (d["cutout"]["width"].as_f64().unwrap() - 26.0).abs() < 1e-9 && (d["cutout"]["height"].as_f64().unwrap() - 16.0).abs() < 1e-9,
        "{}",
        d["cutout"]
    );
    assert_eq!(d["photo"]["regions"], 2);
    assert_eq!(d["damage"]["points"], 4);
    // Light marks on a dark skin.
    let t = run(&mut s, "repair.trace_photo", json!({"data": photo(), "mm_per_pixel": 0.5, "invert": true}));
    assert_eq!(t["regions"].as_array().unwrap().len(), 1);
    assert!(t["regions"][0]["area_mm2"].as_f64().unwrap() > 7000.0);
}

#[test]
fn the_sketch_form_carries_on_into_a_flange_and_its_flat_pattern() {
    let mut s = Session::default();
    let r = run(&mut s, "repair.doubler_sketch", json!({"damage": [[-20, -5], [20, -5], [20, 5], [-20, 5]], "at": [100, 50]}));
    let rivets = r["rivets"].as_u64().unwrap() as usize;
    assert!(rivets >= 8, "{r}");
    assert_eq!(r["curves"].as_u64().unwrap() as usize, 8 + rivets);
    assert!(r["created_sketch"].is_u64());
    run(&mut s, "sketch.finish", json!({}));
    let on_plate = r["profile_point"].clone();
    assert!(
        crate::params::vec2(&on_plate).is_some_and(|q| q.dist(Vec2::new(100.0, 50.0)) > 10.0 && q.dist(Vec2::new(100.0, 50.0)) < 40.0),
        "{on_plate}"
    );
    run(&mut s, "sheet.flange", json!({"sketch": "Doubler", "profiles": [{"point": on_plate}]}));
    let flat = run(&mut s, "sheet.flat_pattern", json!({}));
    assert_eq!(flat["cutouts"].as_array().unwrap().len(), rivets, "{}", flat["flat_size_mm"]);
    let size = flat["flat_size_mm"].as_array().unwrap();
    assert!((size[0].as_f64().unwrap() - 102.0).abs() < 1e-6 && (size[1].as_f64().unwrap() - 72.0).abs() < 1e-6, "{size:?}");
    let dxf = run(&mut s, "sheet.export_dxf", json!({}));
    assert_eq!(dxf_entities(dxf["dxf"].as_str().unwrap()).iter().filter(|(k, l)| k == "CIRCLE" && l == "CUT").count(), rivets);
    // Into an open sketch, and undone as one step.
    let mut s = Session::default();
    run(&mut s, "sketch.create", json!({"plane": "XZ", "name": "Skin"}));
    let id = s.active_sketch.unwrap();
    let r = run(&mut s, "repair.doubler_sketch", json!({"damage": [[0, 0], [10, 0], [10, 10]], "cutout": "circle", "rows": 1}));
    assert!(r["created_sketch"].is_null());
    assert!(r["curves"].as_u64().unwrap() > 3);
    assert!(s.doc.sketch(id).unwrap().curves.len() > 3);
    run(&mut s, "edit.undo", json!({}));
    assert_eq!(s.doc.sketch(id).unwrap().curves.len(), 0);
}

#[test]
fn bad_inputs_are_refused() {
    let mut s = Session::default();
    assert!(fails(&mut s, "repair.doubler", json!({})).contains("damage"));
    assert!(fails(&mut s, "repair.doubler", json!({"damage": [[0, 0], [1, 1]]})).contains("3 points"));
    assert!(fails(&mut s, "repair.doubler", json!({"damage": [[0, 0], [1, 1], [2, 2]]})).contains("no area"));
    assert!(fails(&mut s, "repair.doubler", json!({"damage": [[0, 0], [1, "x"], [2, 2]]})).contains("damage"));
    let ok = json!([[0, 0], [10, 0], [10, 10]]);
    assert!(fails(&mut s, "repair.doubler", json!({"damage": ok, "rows": 0})).contains("rows"));
    assert!(fails(&mut s, "repair.doubler", json!({"damage": ok, "pitch": 1})).contains("pitch"));
    assert!(fails(&mut s, "repair.doubler", json!({"damage": ok, "edge_distance": 1})).contains("edge_distance"));
    assert!(fails(&mut s, "repair.doubler", json!({"damage": ok, "cutout": "hexagon"})).contains("cutout"));
    assert!(fails(&mut s, "repair.doubler", json!({"damage": ok, "rivet_diameter": "x"})).contains("rivet_diameter"));
    assert!(fails(&mut s, "repair.doubler", json!({"damage": [[0, 0], [1e5, 0], [1e5, 1e5]]})).contains("too large"));
    assert!(
        fails(
            &mut s,
            "repair.doubler",
            json!({"damage": ok, "pitch": 8, "rows": 10, "rivet_diameter": 3.2, "row_pitch": 6.4, "cleanup_margin": 1000})
        )
        .contains("5000")
    );
    assert!(fails(&mut s, "repair.doubler", json!({"photo": "x"})).contains("photo"));
    assert!(fails(&mut s, "repair.trace_photo", json!({"data": photo()})).contains("scale"));
    assert!(fails(&mut s, "repair.trace_photo", json!({"data": "@@@", "mm_per_pixel": 1})).contains("base64"));
    assert!(
        fails(&mut s, "repair.trace_photo", json!({"data": solidcraft_doc::canvas::base64_encode(b"plain text"), "mm_per_pixel": 1})).contains("PNG")
    );
    assert!(fails(&mut s, "repair.trace_photo", json!({"data": photo(), "mm_per_pixel": 0.5, "region": 7})).contains("region"));
    assert!(fails(&mut s, "repair.trace_photo", json!({"data": photo(), "mm_per_pixel": 0.5, "at": [1, 1]})).contains("no mark"));
    assert!(
        fails(&mut s, "repair.trace_photo", json!({"data": photo(), "calibrate": {"a": [1, 1], "b": [1, 1], "distance": 5}})).contains("different")
    );
    assert!(fails(&mut s, "repair.trace_photo", json!({"data": photo(), "mm_per_pixel": 0.5, "threshold": 2})).contains("threshold"));
    // A photo with no mark at all.
    let blank = solidcraft_doc::canvas::base64_encode(&gray_png(10, 10, &[255; 100]).unwrap());
    assert!(fails(&mut s, "repair.doubler", json!({"photo": {"data": blank, "mm_per_pixel": 1}})).contains("no mark"));
    assert!(fails(&mut s, "repair.trace_photo", json!({"path": "/nonexistent/photo.png", "mm_per_pixel": 1})).contains("photo.png"));
}

/// Every parameter of the repair commands, every hostile value (the shared fuzz covers only a
/// few of these keys): a refusal, never a panic. `path` writes files and has its own test above.
#[test]
fn hostile_repair_params_never_panic() {
    let keys = [
        "damage",
        "photo",
        "thickness",
        "rivet_diameter",
        "edge_distance",
        "pitch",
        "row_pitch",
        "rows",
        "stagger",
        "cutout",
        "cleanup_margin",
        "corner_radius",
        "angle",
        "notes",
        "plane",
        "name",
        "at",
        "data",
        "threshold",
        "invert",
        "despeckle",
        "mm_per_pixel",
        "width_mm",
        "calibrate",
        "min_area",
        "simplify",
        "origin",
        "region",
    ];
    let base = [
        json!({}),
        json!({"damage": [[0, 0], [10, 0], [10, 10]]}),
        json!({"photo": {"data": photo(), "mm_per_pixel": 0.5}}),
        json!({"data": photo(), "mm_per_pixel": 0.5}),
    ];
    for id in ["repair.doubler", "repair.doubler_sketch", "repair.trace_photo"] {
        for v in crate::tests::hostile_values() {
            for k in keys {
                for b in &base {
                    let mut p = b.clone();
                    p[k] = v.clone();
                    let mut s = Session::default();
                    if let Err(EngineError::Internal(_, m)) = s.execute(id, &p) {
                        panic!("{id} {p}: {m}");
                    }
                    // Nested in `photo` and `calibrate` too.
                    let mut p = b.clone();
                    p["photo"] = json!({k: v.clone(), "data": photo(), "mm_per_pixel": 0.5});
                    p["calibrate"] = json!({k: v.clone()});
                    if let Err(EngineError::Internal(_, m)) = s.execute(id, &p) {
                        panic!("{id} {p}: {m}");
                    }
                }
            }
        }
    }
}

#[test]
fn a_rounded_cleanout_never_cuts_into_the_damage() {
    let mut s = Session::default();
    // No margin and the default 6 mm corners on a 10 × 10 square: the plain bounding box would lose the corners.
    let square = json!([[0, 0], [10, 0], [10, 10], [0, 10]]);
    let d = run(&mut s, "repair.doubler", json!({"damage": square, "cleanup_margin": 0, "rows": 1}));
    let cutout = shape_loop(&d["cutout"]);
    // With no margin the corners sit on the trim line itself (as they would on the plain box).
    let inside = |lp: &Loop2, p: Vec2| lp.contains(p) || dist_to_loop(lp, p) < 1e-6;
    for c in [Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0), Vec2::new(0.0, 10.0)] {
        assert!(inside(&cutout, c), "corner {c:?} outside {}", d["cutout"]);
    }
    // 6 mm corners on a 10 mm box make it a circle of radius 5, grown to reach the corners: 10√2 across.
    assert!((d["cutout"]["width"].as_f64().unwrap() - 10.0 * 2f64.sqrt()).abs() < 1e-9, "{}", d["cutout"]);
    // A large radius with a margin, on a tilted outline: every damage point stays inside.
    let (sn, cs) = 20f64.to_radians().sin_cos();
    let rot = |x: f64, y: f64| json!([x * cs - y * sn, x * sn + y * cs]);
    let damage = json!([rot(-30.0, -8.0), rot(30.0, -8.0), rot(30.0, 8.0), rot(-30.0, 8.0), rot(0.0, 10.0)]);
    let d = run(&mut s, "repair.doubler", json!({"damage": damage, "cleanup_margin": 2, "corner_radius": 40, "rows": 1}));
    let cutout = shape_loop(&d["cutout"]);
    for p in pts(&damage) {
        assert!(inside(&cutout, p), "{p:?} outside {}", d["cutout"]);
    }
    // And without rounding the box is unchanged.
    let d = run(&mut s, "repair.doubler", json!({"damage": square, "cleanup_margin": 0, "corner_radius": 0, "rows": 1}));
    assert!((d["cutout"]["width"].as_f64().unwrap() - 10.0).abs() < 1e-9);
}

#[test]
fn rivets_on_a_curved_row_never_overlap() {
    let mut s = Session::default();
    // Around a circle the straight distance between neighbours is shorter than the developed spacing.
    for pitch in [6.4, 7.0, 8.0, 12.0] {
        let p = json!({"damage": [[0, 0], [10, 0], [10, 10], [0, 10]], "cutout": "circle", "rows": 2, "rivet_diameter": 3.2, "pitch": pitch, "row_pitch": 6.4});
        match s.execute("repair.doubler", &p) {
            Ok(d) => {
                let rivets = pts(&d["rivets"]);
                for (i, a) in rivets.iter().enumerate() {
                    for b in rivets.iter().skip(i + 1) {
                        assert!(a.dist(*b) >= 6.4 - 1e-9, "pitch {pitch}: {a:?} {b:?}");
                    }
                }
                for r in d["rows"].as_array().unwrap() {
                    assert!(r["pitch"].as_f64().unwrap() <= pitch + 1e-9);
                }
            }
            Err(EngineError::Internal(_, m)) => panic!("{m}"),
            Err(e) => assert!(e.to_string().contains("pitch"), "{e}"),
        }
    }
    // The tightest allowed pitch on a round row cannot be met: refused, not overlapped.
    assert!(
        fails(
            &mut s,
            "repair.doubler",
            json!({"damage": [[0, 0], [10, 0], [10, 10], [0, 10]], "cutout": "circle", "rows": 1, "rivet_diameter": 3.2, "pitch": 6.4})
        )
        .contains("pitch")
    );
}

#[test]
fn a_phone_sized_photo_keeps_its_own_pixel_coordinates() {
    // 8200 × 10 px, a 600 px wide black block from x = 100: traced 3 × coarser, but every input
    // and output is in the photo's pixels.
    let (w, h) = (8200u32, 10u32);
    let mut luma = vec![255u8; (w * h) as usize];
    for y in 0..h {
        for x in 100..700 {
            luma[(y * w + x) as usize] = 0;
        }
    }
    let data = solidcraft_doc::canvas::base64_encode(&gray_png(w, h, &luma).unwrap());
    let mut s = Session::default();
    // Two pixels 8200 apart are 820 mm apart: 0.1 mm per photo pixel.
    let t = run(
        &mut s,
        "repair.trace_photo",
        json!({"data": data, "calibrate": {"a": [0, 5], "b": [8200, 5], "distance": 820}, "origin": "image", "despeckle": false}),
    );
    assert_eq!((t["width_px"].as_u64(), t["height_px"].as_u64(), t["downsampled_by"].as_u64()), (Some(8200), Some(10), Some(3)));
    assert!((t["mm_per_pixel"].as_f64().unwrap() - 0.1).abs() < 1e-12);
    let r = &t["regions"][0];
    // The block is 60 mm wide starting 10 mm in (to within one coarse pixel, 0.3 mm).
    let bbox = r["bbox"].as_array().unwrap();
    assert!((bbox[0].as_f64().unwrap() - 10.0).abs() < 0.31 && (bbox[2].as_f64().unwrap() - 70.0).abs() < 0.31, "{bbox:?}");
    // Where the mark is, in the photo's pixels.
    let bpx = r["bbox_px"].as_array().unwrap();
    assert!((bpx[0].as_f64().unwrap() - 99.0).abs() < 1e-9 && (bpx[2].as_f64().unwrap() - 699.0).abs() < 1e-9, "{bpx:?}");
    assert!((r["center_px"][0].as_f64().unwrap() - 399.0).abs() < 1e-9);
    // `at` in photo pixels finds it.
    assert_eq!(run(&mut s, "repair.trace_photo", json!({"data": data, "mm_per_pixel": 0.1, "at": [400, 5], "despeckle": false}))["chosen"], 0);
    assert!(fails(&mut s, "repair.trace_photo", json!({"data": data, "mm_per_pixel": 0.1, "at": [4000, 5], "despeckle": false})).contains("no mark"));
}

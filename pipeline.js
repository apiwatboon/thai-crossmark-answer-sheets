/* pipeline.js — พอร์ตของ run_phase2.py + grader_core.py มาเป็น JavaScript (เบราว์เซอร์)
 * ใช้ OpenCV.js (global `cv`) + onnxruntime-web (global `ort`)
 * ขอบเขต v1: pose = YOLO26 e2e (output 1x300x18), crop = grid, reader = pixel_count
 * ค่าคงที่/ลำดับขั้นตรงกับ Python ที่ล็อกไว้ (shift0.14 + inset60% + rx robust_guard)
 */

const CFG = {
  S: 640,
  N_CHOICES: 5, CELL: 72, EXPAND_CELLS: 0.5,
  COL_ROWS: { 0: 15, 1: 20, 2: 25, 3: 30 },
  BOT_SHIFT_CELLS: 0.14,
  DEDUP_FRAC: 0.4,
  OPT_NAMES: ["A", "B", "C", "D", "E"],
  PIXEL_THRESHOLD: 128, MIN_FILL_RATIO: 0.05, CX_TOL: 0.12,
  CELL_INSET_FRAC: 0.2,
  CLAHE_CLIP: 2.0, CLAHE_TILE: 16,
  GREENNESS_DELTA: 4,
  POSE_CONF: 0.30,            // กรองดีเทกชันจาก e2e (เทียบ ultralytics ~0.25-0.3)
};
const CLASS_LABEL = { 0: 15, 1: 20, 2: 25, 3: 30 };

// ── ensemble: คู่ (pose × reader) ที่ดีที่สุด เรียงตามความแม่นยำ + โหวตเสียงข้างมาก ─
//   pose top-3 (OKS) = yolo26n/yolov8s/yolo11s_pose ; reader = pixel_count + bbox top-3 (F1)
const ENSEMBLE = {
  POSE: ["yolo26n_pose", "yolov8s_pose", "yolo11s_pose"],    // decoder: yolo26=e2e, v8/v11=raw
  BBOX: [                                                    // conf จาก runs_cross/f1_threshold_best.csv
    { name: "yolo11s_bbox", conf: 0.33 },                   // raw (1,5,8400) — F1 0.998
    { name: "yolo26n_bbox", conf: 0.29 },                   // e2e (1,300,6)
    { name: "yolov8s_bbox", conf: 0.49 },                   // raw (1,5,8400)
  ],
  DETECT_FLOOR: 0.05,   // floor ตอน decode ; ตัดจริงด้วย conf ราย bbox ใน bboxAnswers
};
// รายการคู่ (pose, reader) เรียงจากดีที่สุด → แย่สุด (q_acc บนชุด val, expand-crop)
//   ที่มา: results/phase2_cross/summary.csv (crop=grid) ; คู่ที่ใช้ yolov8s_bbox ไม่มีในตาราง
//   จึงจัดไว้ท้ายสุดตามคุณภาพ pose ( tier 1/3/6 = 6 คู่แรกล้วนมีข้อมูลรองรับ )
const PAIRS = [
  { pose: "yolo26n_pose", reader: "pixel_count", acc: 0.8518 },
  { pose: "yolo26n_pose", reader: "yolo11s_bbox", acc: 0.8509 },
  { pose: "yolo26n_pose", reader: "yolo26n_bbox", acc: 0.8504 },
  { pose: "yolov8s_pose", reader: "yolo11s_bbox", acc: 0.8491 },
  { pose: "yolov8s_pose", reader: "pixel_count", acc: 0.8489 },
  { pose: "yolov8s_pose", reader: "yolo26n_bbox", acc: 0.8489 },
  { pose: "yolo11s_pose", reader: "pixel_count", acc: 0.8322 },
  { pose: "yolo11s_pose", reader: "yolo26n_bbox", acc: 0.8319 },
  { pose: "yolo11s_pose", reader: "yolo11s_bbox", acc: 0.8307 },
  { pose: "yolo26n_pose", reader: "yolov8s_bbox", acc: 0.8470 },
  { pose: "yolov8s_pose", reader: "yolov8s_bbox", acc: 0.8450 },
  { pose: "yolo11s_pose", reader: "yolov8s_bbox", acc: 0.8280 },
];
// ระดับการโหวต = จำนวนคู่ที่ดีที่สุด N คู่ ; quorum = เสียงข้างมาก ⌊N/2⌋+1
const TIERS = { 1: 1, 3: 3, 6: 6, 9: 9, 12: 12 };
function tierQuorum(level) { const n = TIERS[level] || 12; return (n >> 1) + 1; }

// ── พีชคณิตเส้น: fit x = a*y + b (least squares deg-1) ────────────────────────
function polyfit1(xv, yv) {
  // คืน [a, b] ของ yv = a*xv + b  (เลียน np.polyfit(xv, yv, 1))
  const n = xv.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += xv[i]; sy += yv[i]; sxx += xv[i] * xv[i]; sxy += xv[i] * yv[i]; }
  const den = n * sxx - sx * sx;
  if (Math.abs(den) < 1e-9) return [0, sy / n];
  const a = (n * sxy - sx * sy) / den;
  const b = (sy - a * sx) / n;
  return [a, b];
}
function median(arr) {
  const a = [...arr].sort((p, q) => p - q); const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

// ── pose: letterbox + decode (YOLO26 e2e) ────────────────────────────────────
function letterbox(srcMat) {
  const S = CFG.S, w = srcMat.cols, h = srcMat.rows;
  const r = Math.min(S / w, S / h);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const dw = (S - nw) / 2, dh = (S - nh) / 2;
  const top = Math.round(dh - 0.1), bottom = Math.round(dh + 0.1);
  const left = Math.round(dw - 0.1), right = Math.round(dw + 0.1);
  const rs = new cv.Mat();
  cv.resize(srcMat, rs, new cv.Size(nw, nh), 0, 0, cv.INTER_LINEAR);
  const out = new cv.Mat();
  cv.copyMakeBorder(rs, out, top, bottom, left, right, cv.BORDER_CONSTANT, new cv.Scalar(114, 114, 114, 255));
  rs.delete();
  return { lb: out, r, padx: left, pady: top };
}

function matToTensor(lbMat) {
  // lbMat = RGBA หรือ BGR? ที่นี่ lbMat มาจาก cv (BGR). แปลง BGR->RGB, /255, CHW
  const S = CFG.S;
  const rgb = new cv.Mat();
  cv.cvtColor(lbMat, rgb, cv.COLOR_BGR2RGB);
  const data = rgb.data;            // length S*S*3, ลำดับ HWC
  const out = new Float32Array(3 * S * S);
  const plane = S * S;
  for (let i = 0; i < plane; i++) {
    out[i] = data[i * 3] / 255;             // R
    out[plane + i] = data[i * 3 + 1] / 255; // G
    out[2 * plane + i] = data[i * 3 + 2] / 255; // B
  }
  rgb.delete();
  return new ort.Tensor("float32", out, [1, 3, S, S]);
}

function iouBox(a, b) {
  const xx1 = Math.max(a.x1, b.x1), yy1 = Math.max(a.y1, b.y1);
  const xx2 = Math.min(a.x2, b.x2), yy2 = Math.min(a.y2, b.y2);
  const w = Math.max(0, xx2 - xx1), h = Math.max(0, yy2 - yy1), inter = w * h;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1), areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-9);
}

function nms(cand, iouThr = 0.7) {
  const order = cand.map((_, i) => i).sort((p, q) => cand[q].conf - cand[p].conf);
  const supp = new Array(cand.length).fill(false), keep = [];
  for (let i = 0; i < order.length; i++) {
    const ci = order[i];
    if (supp[ci]) continue;
    keep.push(ci);
    for (let j = i + 1; j < order.length; j++) {
      const cj = order[j];
      if (!supp[cj] && iouBox(cand[ci], cand[cj]) > iouThr) supp[cj] = true;
    }
  }
  return keep.map(i => cand[i]);
}

// decode สำหรับ YOLOv8/YOLO11 pose (เอาต์พุตดิบ 1x20x8400) — ต้องทำ NMS เอง
// ช่อง: [0:4]=box(cx,cy,w,h) [4:4+nc]=คลาส [4+nc:]=keypoint*(x,y,vis), ทุกค่าเป็นพิกัด input 640
function decodeRaw(out, dims, geo, nc = 4, nkpt = 4) {
  const A = dims[2], base = 4 + nc;
  const at = (c, a) => out[c * A + a];
  const un = (x, y) => [(x - geo.padx) / geo.r, (y - geo.pady) / geo.r];
  const cand = [];
  for (let a = 0; a < A; a++) {
    let bestC = 0, bestS = at(4, a);
    for (let k = 1; k < nc; k++) { const s = at(4 + k, a); if (s > bestS) { bestS = s; bestC = k; } }
    if (bestS < CFG.POSE_CONF) continue;
    const cx = at(0, a), cy = at(1, a), w = at(2, a), h = at(3, a);
    const tl = un(at(base, a), at(base + 1, a));
    const tr = un(at(base + 3, a), at(base + 4, a));
    const bl = un(at(base + 6, a), at(base + 7, a));
    const br = un(at(base + 9, a), at(base + 10, a));
    cand.push({
      cls: bestC, conf: bestS,
      cx: (cx - geo.padx) / geo.r, cy: (cy - geo.pady) / geo.r,
      x1: (cx - w / 2 - geo.padx) / geo.r, y1: (cy - h / 2 - geo.pady) / geo.r,
      x2: (cx + w / 2 - geo.padx) / geo.r, y2: (cy + h / 2 - geo.pady) / geo.r,
      tl, tr, bl, br,
    });
  }
  return nms(cand, 0.7);
}

function decodeE2E(out, dims, geo) {
  // out: Float32Array, dims = [1,300,18]; geo = {r,padx,pady}
  const n = dims[1], step = dims[2];
  const dets = [];
  for (let i = 0; i < n; i++) {
    const o = i * step;
    const conf = out[o + 4];
    if (conf < CFG.POSE_CONF) continue;
    const cls = Math.round(out[o + 5]);
    const un = (x, y) => [(x - geo.padx) / geo.r, (y - geo.pady) / geo.r];
    const tl = un(out[o + 6], out[o + 7]);
    const tr = un(out[o + 9], out[o + 10]);
    const bl = un(out[o + 12], out[o + 13]);
    const br = un(out[o + 15], out[o + 16]);
    const cx = ((out[o] + out[o + 2]) / 2 - geo.padx) / geo.r;
    const cy = ((out[o + 1] + out[o + 3]) / 2 - geo.pady) / geo.r;
    dets.push({ cls, conf, cx, cy, tl, tr, bl, br });
  }
  return dets;
}

// ── bbox detector decode (reader=yolo_bbox) — ไม่มี keypoint, คืน {cx,cy,conf} พิกัด crop ──
// yolo26 bbox: e2e (1,300,6)=[x1,y1,x2,y2,conf,cls] ; v8/v11 bbox: raw (1,5,8400) ต้อง NMS
function decodeDetectE2E(out, dims, geo, minConf) {
  const n = dims[1], step = dims[2], dets = [];
  for (let i = 0; i < n; i++) {
    const o = i * step, conf = out[o + 4];
    if (conf < minConf) continue;
    dets.push({
      conf,
      cx: ((out[o] + out[o + 2]) / 2 - geo.padx) / geo.r,
      cy: ((out[o + 1] + out[o + 3]) / 2 - geo.pady) / geo.r,
    });
  }
  return dets;
}

function decodeDetectRaw(out, dims, geo, minConf) {
  const A = dims[2], nc = dims[1] - 4, at = (c, a) => out[c * A + a], cand = [];
  for (let a = 0; a < A; a++) {
    let bestS = at(4, a);
    for (let k = 1; k < nc; k++) { const s = at(4 + k, a); if (s > bestS) bestS = s; }
    if (bestS < minConf) continue;
    const cx = at(0, a), cy = at(1, a), w = at(2, a), h = at(3, a);
    cand.push({
      conf: bestS,
      cx: (cx - geo.padx) / geo.r, cy: (cy - geo.pady) / geo.r,
      x1: (cx - w / 2 - geo.padx) / geo.r, y1: (cy - h / 2 - geo.pady) / geo.r,
      x2: (cx + w / 2 - geo.padx) / geo.r, y2: (cy + h / 2 - geo.pady) / geo.r,
    });
  }
  return nms(cand, 0.7);
}

function snapRow(cy, rowC) {
  let bi = 0, bd = Infinity;
  for (let i = 0; i < rowC.length; i++) { const d = Math.abs(rowC[i] - cy); if (d < bd) { bd = d; bi = i; } }
  return bi;
}
function snapOption(cx, optC, w) {
  let best = null, bd = Infinity;
  for (const o of CFG.OPT_NAMES) { const d = Math.abs(optC[o] - cx); if (d < bd) { bd = d; best = o; } }
  return bd > w * CFG.CX_TOL ? "?" : best;
}

// port ของ run_phase2.yolo_bbox_answers: ดีเทกชันบน crop -> คำตอบรายแถว
function bboxAnswers(dets, cls, rowC, optC, w, confThr) {
  const rpc = CFG.COL_ROWS[cls];
  const cb = new Map();   // "ri,opt" -> conf สูงสุด
  for (const d of dets) {
    if (d.conf < confThr) continue;
    const ri = snapRow(d.cy, rowC);
    const opt = snapOption(d.cx, optC, w);
    if (opt === "?") continue;
    const k = ri + "," + opt;
    if (!cb.has(k) || d.conf > cb.get(k)) cb.set(k, d.conf);
  }
  const rb = new Map();   // ri -> [opt, conf]
  for (const [k, c] of cb) {
    const ri = +k.split(",")[0], opt = k.split(",")[1];
    if (!rb.has(ri) || c > rb.get(ri)[1]) rb.set(ri, [opt, c]);
  }
  const ans = [];
  for (let ri = 0; ri < rpc; ri++) ans.push([ri + 1, rb.has(ri) ? rb.get(ri)[0] : "None"]);
  return ans;
}

// ── dedup + ordering (ตรง run_phase2.py) ─────────────────────────────────────
function norm(a) { return Math.hypot(a[0], a[1]); }
function sub(a, b) { return [a[0] - b[0], a[1] - b[1]]; }

function dedupCols(yo, frac = CFG.DEDUP_FRAC) {
  if (yo.length <= 1) return yo;
  const widths = yo.map(y => norm(sub(y.tr, y.tl)));
  const thr = frac * median(widths);
  const sorted = [...yo].sort((a, b) => a.cx - b.cx);
  const out = [];
  for (const y of sorted) {
    if (out.length && Math.abs(y.cx - out[out.length - 1].cx) < thr) {
      if ((y.conf || 0) > (out[out.length - 1].conf || 0)) out[out.length - 1] = y;
    } else out.push(y);
  }
  return out;
}

function getPhysicalOrder(o) {
  const n = o.length;
  if (n <= 1) return [...Array(n).keys()];
  const c = o.map(x => [(x.tl[0] + x.tr[0] + x.bl[0] + x.br[0]) / 4,
                        (x.tl[1] + x.tr[1] + x.bl[1] + x.br[1]) / 4]);
  let rv = [0, 0];
  for (const x of o) {
    rv[0] += (x.tr[0] - x.tl[0]) + (x.br[0] - x.bl[0]);
    rv[1] += (x.tr[1] - x.tl[1]) + (x.br[1] - x.bl[1]);
  }
  rv = [rv[0] / n, rv[1] / n];
  const nr = norm(rv);
  const idx = [...Array(n).keys()];
  if (nr < 1e-6) return idx.sort((i, j) => c[i][0] - c[j][0]);
  rv = [rv[0] / nr, rv[1] / nr];
  return idx.sort((i, j) => (c[i][0] * rv[0] + c[i][1] * rv[1]) - (c[j][0] * rv[0] + c[j][1] * rv[1]));
}

function orderColumns(yo) {
  yo = dedupCols(yo);
  const order = getPhysicalOrder(yo);
  return order.map((oi, pi) => [pi, yo[oi]]);  // [(col_idx, obj), ...]
}

// ── crop helpers ─────────────────────────────────────────────────────────────
function expandKeypoints(tl, tr, bl, br, mlr, mtb) {
  const u = v => { const nn = norm(v); return nn > 1e-6 ? [v[0] / nn, v[1] / nn] : v; };
  const tv = u(sub(tr, tl)), bv = u(sub(br, bl)), lv = u(sub(bl, tl)), rv = u(sub(br, tr));
  const tn = norm(sub(tr, tl)), bn = norm(sub(br, bl)), ln = norm(sub(bl, tl)), rn = norm(sub(br, tr));
  return [
    [tl[0] - tv[0] * tn * mlr - lv[0] * ln * mtb, tl[1] - tv[1] * tn * mlr - lv[1] * ln * mtb],
    [tr[0] + tv[0] * tn * mlr - rv[0] * rn * mtb, tr[1] + tv[1] * tn * mlr - rv[1] * rn * mtb],
    [bl[0] - bv[0] * bn * mlr + lv[0] * ln * mtb, bl[1] - bv[1] * bn * mlr + lv[1] * ln * mtb],
    [br[0] + bv[0] * bn * mlr + rv[0] * rn * mtb, br[1] + bv[1] * bn * mlr + rv[1] * rn * mtb],
  ];
}

function perspectiveCrop(srcMat, tl, tr, bl, br, ow, oh) {
  const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [tl[0], tl[1], tr[0], tr[1], bl[0], bl[1], br[0], br[1]]);
  const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, ow, 0, 0, oh, ow, oh]);
  const M = cv.getPerspectiveTransform(srcTri, dstTri);
  const dst = new cv.Mat();
  cv.warpPerspective(srcMat, dst, M, new cv.Size(ow, oh), cv.INTER_LANCZOS4,
    cv.BORDER_CONSTANT, new cv.Scalar(255, 255, 255, 255));
  srcTri.delete(); dstTri.delete(); M.delete();
  return dst;
}

function greenMask(bgr) {
  const hsv = new cv.Mat();
  cv.cvtColor(bgr, hsv, cv.COLOR_BGR2HSV);
  const lo = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [35, 40, 30, 0]);
  const hi = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [90, 255, 255, 0]);
  const m = new cv.Mat();
  cv.inRange(hsv, lo, hi, m);
  const k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
  cv.morphologyEx(m, m, cv.MORPH_CLOSE, k);
  hsv.delete(); lo.delete(); hi.delete(); k.delete();
  return m;  // CV_8UC1
}

function greenMaskSensitive(bgr, delta = CFG.GREENNESS_DELTA) {
  const ch = new cv.MatVector();
  cv.split(bgr, ch);
  const b = ch.get(0), g = ch.get(1), r = ch.get(2);
  const m = new cv.Mat(bgr.rows, bgr.cols, cv.CV_8UC1);
  const bd = b.data, gd = g.data, rd = r.data, md = m.data;
  for (let i = 0; i < md.length; i++) {
    md[i] = (gd[i] - Math.max(rd[i], bd[i]) >= delta) ? 255 : 0;
  }
  ch.delete();
  const close = (sz) => { const k = cv.getStructuringElement(cv.MORPH_RECT, sz); cv.morphologyEx(m, m, cv.MORPH_CLOSE, k); k.delete(); };
  close(new cv.Size(5, 5)); close(new cv.Size(3, 60)); close(new cv.Size(30, 40));
  return m;
}

function maskAt(mask, y, x) { return mask.data[y * mask.cols + x]; }

function fitEdgeLine(mask, y0, y1, x0, x1, side) {
  const xs = [], ys = [];
  for (let y = y0; y < y1; y++) {
    let found = -1;
    if (side === "right") { for (let x = x1 - 1; x >= x0; x--) if (maskAt(mask, y, x) > 0) { found = x; break; } }
    else { for (let x = x0; x < x1; x++) if (maskAt(mask, y, x) > 0) { found = x; break; } }
    // นับว่ามี >=3 จุดเขียวในแถบหรือไม่ (เลียน len(gp)<3)
    let cnt = 0; for (let x = x0; x < x1; x++) if (maskAt(mask, y, x) > 0) cnt++;
    if (cnt < 3 || found < 0) continue;
    xs.push(found); ys.push(y);
  }
  if (xs.length < 20) return null;
  return polyfit1(ys, xs);  // x = a*y + b
}

function colSums(mask, st, sb) {
  // คืน fraction รายคอลัมน์ x: sum(mask[st:h-sb, x])/((h-st-sb)*255)
  const h = mask.rows, w = mask.cols, denom = (h - st - sb) * 255;
  const cs = new Float64Array(w);
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = st; y < h - sb; y++) s += maskAt(mask, y, x);
    cs[x] = s / denom;
  }
  return cs;
}

function findLxRx(s2, isLast) {
  const mask = greenMask(s2);
  const h = s2.rows, w = s2.cols;
  const st = Math.floor(h * 0.15), sb = Math.floor(h * 0.15), lw = Math.floor(w * 0.12);
  const cs = colSums(mask, st, sb);
  let lx = null, rx = null;
  let anyL = false; for (let x = 0; x < lw; x++) if (cs[x] > 0.08) { anyL = true; break; }
  if (anyL) lx = fitEdgeLine(mask, st, h - sb, 0, lw, "right");

  if (isLast) {
    const ms = greenMaskSensitive(s2);
    const st2 = Math.floor(h * 0.02), x0 = Math.floor(w * 0.80);
    const xs = [], ys = [];
    for (let y = st2; y < h - st2; y++) {
      if (maskAt(ms, y, w - 1) > 0) continue;
      for (let x = w - 1; x >= x0; x--) if (maskAt(ms, y, x) > 0) { xs.push(x); ys.push(y); break; }
    }
    if (xs.length >= 20) {
      const scanned = (h - st2) - st2;
      const s = lx ? lx[0] : 0.0;
      const med = median(xs);
      const keepX = [], keepY = []; let keepCnt = 0;
      for (let i = 0; i < xs.length; i++) if (Math.abs(xs[i] - med) <= 12) { keepX.push(xs[i]); keepY.push(ys[i]); keepCnt++; }
      if (xs.length >= 0.6 * scanned && keepCnt >= 20) {
        rx = polyfit1(keepY, keepX);
      } else {
        const ints = xs.map((x, i) => x - s * ys[i]);
        rx = [s, median(ints)];
      }
    }
    ms.delete();
  } else {
    const rw = Math.floor(w * 0.12);
    let anyR = false; for (let x = w - rw; x < w; x++) if (cs[x] > 0.08) { anyR = true; break; }
    if (anyR) rx = fitEdgeLine(mask, st, h - sb, w - rw, w, "left");
  }
  mask.delete();
  return [lx, rx];
}

function detectTopLine(s2, covThr = 0.45) {
  const mask = greenMask(s2);
  const h = s2.rows, w = s2.cols, x0 = Math.floor(w * 0.20), x1 = Math.floor(w * 0.80);
  const denom = (x1 - x0) * 255;
  const cov = y => { let s = 0; for (let x = x0; x < x1; x++) s += maskAt(mask, y, x); return s / denom; };
  const band = [];
  for (let y = Math.floor(h * 0.005); y < Math.floor(h / 3); y++) {
    if (cov(y) > covThr) band.push(y); else if (band.length) break;
  }
  if (!band.length) { mask.delete(); return null; }
  const yb = band[band.length - 1];
  const xs = [], ys = [];
  for (let x = x0; x < x1; x += 3) {
    let last = -1, cnt = 0;
    for (let y = 0; y <= Math.min(yb + 30, h - 1); y++) if (maskAt(mask, y, x) > 0) { last = y; cnt++; }
    if (cnt >= 3) { xs.push(x); ys.push(last); }
  }
  mask.delete();
  if (ys.length < 20) return null;
  return polyfit1(xs, ys);  // y = a*x + b
}

function detectTopPadgreen(s2, padcells = 1.0) {
  const pad = Math.floor(padcells * CFG.CELL), w = s2.cols;
  const strip = new cv.Mat(pad, w, s2.type(), new cv.Scalar(0, 255, 0, 255));
  const stacked = new cv.Mat();
  const vv = new cv.MatVector(); vv.push_back(strip); vv.push_back(s2);
  cv.vconcat(vv, stacked);
  const tc = detectTopLine(stacked);
  strip.delete(); stacked.delete(); vv.delete();
  return tc === null ? null : [tc[0], tc[1] - pad];
}

function detectBottomLine(s2, covThr = 0.45) {
  const mask = greenMask(s2);
  const h = s2.rows, w = s2.cols, x0 = Math.floor(w * 0.20), x1 = Math.floor(w * 0.80);
  const denom = (x1 - x0) * 255;
  const cov = y => { let s = 0; for (let x = x0; x < x1; x++) s += maskAt(mask, y, x); return s / denom; };
  const band = [];
  for (let y = h - 1; y > Math.floor(h / 2); y--) {
    if (cov(y) > covThr) band.push(y); else if (band.length) break;
  }
  if (!band.length) { mask.delete(); return null; }
  const yt = band[band.length - 1];
  const xs = [], ys = [];
  for (let x = x0; x < x1; x += 3) {
    const yOff = Math.max(0, yt - 30);
    let first = -1, cnt = 0;
    for (let y = yOff; y < h; y++) if (maskAt(mask, y, x) > 0) { if (first < 0) first = y; cnt++; }
    if (cnt >= 3) { xs.push(x); ys.push(first); }
  }
  mask.delete();
  if (ys.length < 20) return null;
  return polyfit1(xs, ys);
}

function isect(v, hh) {
  // v: x = a*y + b ; hh: y = c*x + d
  const a = v[0], b = v[1], c = hh[0], d = hh[1], den = 1 - c * a;
  if (Math.abs(den) < 1e-9) return null;
  const y = (c * b + d) / den;
  return [a * y + b, y];
}

function cropGrid(srcMat, obj, nr, isLast) {
  const { CELL, N_CHOICES, EXPAND_CELLS, BOT_SHIFT_CELLS } = CFG;
  const te = expandKeypoints(obj.tl, obj.tr, obj.bl, obj.br, EXPAND_CELLS / N_CHOICES, EXPAND_CELLS / nr);
  const ow = Math.floor((N_CHOICES + 2 * EXPAND_CELLS) * CELL);
  const oh = Math.floor((nr + 2 * EXPAND_CELLS) * CELL);
  const s2 = perspectiveCrop(srcMat, te[0], te[1], te[2], te[3], ow, oh);
  let [lx, rx] = findLxRx(s2, isLast);
  let top = detectTopPadgreen(s2);
  if (top === null) top = [0.0, EXPAND_CELLS * CELL];
  if (lx === null) lx = [0.0, EXPAND_CELLS * CELL];
  if (rx === null) rx = [lx[0], lx[1] + N_CHOICES * CELL];
  const bot = [0.0, EXPAND_CELLS * CELL + nr * CELL + BOT_SHIFT_CELLS * CELL];
  const TL = isect(lx, top), TR = isect(rx, top), BL = isect(lx, bot), BR = isect(rx, bot);
  if ([TL, TR, BL, BR].some(p => p === null)) {
    s2.delete();
    return perspectiveCrop(srcMat, te[0], te[1], te[2], te[3], N_CHOICES * CELL, nr * CELL);
  }
  const out = perspectiveCrop(s2, TL, TR, BL, BR, N_CHOICES * CELL, nr * CELL);
  s2.delete();
  return out;
}

// ── reader: pixel_count ──────────────────────────────────────────────────────
function fixedCenters(nr) {
  const opt = {}; CFG.OPT_NAMES.forEach((o, i) => opt[o] = (i + 0.5) * CFG.CELL);
  const row = []; for (let r = 0; r < nr; r++) row.push((r + 0.5) * CFG.CELL);
  return { opt, row };
}

function pixelCountAnswers(cimg, cls, rowC, optC) {
  const rpc = CFG.COL_ROWS[cls], h = cimg.rows, w = cimg.cols;
  const gray = new cv.Mat();
  cv.cvtColor(cimg, gray, cv.COLOR_BGR2GRAY);
  const clahe = new cv.CLAHE(CFG.CLAHE_CLIP, new cv.Size(CFG.CLAHE_TILE, CFG.CLAHE_TILE));
  clahe.apply(gray, gray);
  const bw = new cv.Mat();
  cv.threshold(gray, bw, CFG.PIXEL_THRESHOLD, 255, cv.THRESH_BINARY_INV);
  const bd = bw.data;

  const re = [0];
  for (let i = 0; i < rowC.length - 1; i++) re.push(Math.floor((rowC[i] + rowC[i + 1]) / 2));
  re.push(h);
  const ov = CFG.OPT_NAMES.slice(0, 5).map(o => optC[o]);
  const oe = [0];
  for (let i = 0; i < 4; i++) oe.push(Math.floor((ov[i] + ov[i + 1]) / 2));
  oe.push(w);
  const ins = Math.round(CFG.CELL_INSET_FRAC * CFG.CELL);

  const ans = [];
  for (let ri = 0; ri < rpc; ri++) {
    const y1 = Math.max(0, re[ri] + ins), y2 = Math.min(h, re[ri + 1] - ins);
    const sc = [];
    for (let oi = 0; oi < 5; oi++) {
      const x1 = Math.max(0, oe[oi] + ins), x2 = Math.min(w, oe[oi + 1] - ins);
      let s = 0, cnt = 0;
      for (let y = y1; y < y2; y++) { const base = y * w; for (let x = x1; x < x2; x++) { s += bd[base + x]; cnt++; } }
      sc.push(cnt > 0 ? s / (cnt * 255) : 0);
    }
    let mx = -1, mi = 0; for (let i = 0; i < 5; i++) if (sc[i] > mx) { mx = sc[i]; mi = i; }
    let nFilled = 0; for (let i = 0; i < 5; i++) if (sc[i] >= CFG.MIN_FILL_RATIO) nFilled++;
    ans.push([ri + 1, mx < CFG.MIN_FILL_RATIO ? "None" : CFG.OPT_NAMES[mi], nFilled]);
  }
  gray.delete(); clahe.delete(); bw.delete();
  return ans;
}

// ── inference helpers (ใช้ร่วม WebGrader + EnsembleGrader) ────────────────────
async function runSession(session, srcMat) {
  const { lb, r, padx, pady } = letterbox(srcMat);
  const tensor = matToTensor(lb);
  lb.delete();
  const feeds = {}; feeds[session.inputNames[0]] = tensor;
  const res = await session.run(feeds);
  const out = res[session.outputNames[0]];
  return { out, geo: { r, padx, pady } };
}
async function runPose(session, srcMat) {
  const { out, geo } = await runSession(session, srcMat);
  // e2e (YOLO26): dims=[1,300,18] → กรอง ; raw (v8/v11): dims=[1,20,8400] → NMS
  return out.dims[2] === 18 ? decodeE2E(out.data, out.dims, geo) : decodeRaw(out.data, out.dims, geo);
}
async function runDetect(session, cropMat, minConf) {
  const { out, geo } = await runSession(session, cropMat);
  // e2e bbox: dims=[1,300,6] ; raw bbox: dims=[1,5,8400]
  return out.dims[2] === 6 ? decodeDetectE2E(out.data, out.dims, geo, minConf)
                           : decodeDetectRaw(out.data, out.dims, geo, minConf);
}

// yield ให้ event loop ประมวลผลคลิก "หยุด" (setTimeout = macrotask) แล้วเช็ค abort
async function checkAbort(signal) {
  if (!signal) return;
  await new Promise(r => setTimeout(r));
  if (signal.aborted) throw new DOMException("ยกเลิกการประมวลผล", "AbortError");
}

// ── Grader โมเดลเดียว (ใช้ในเทส/พอร์ตเดิม) ───────────────────────────────────
class WebGrader {
  constructor(session) { this.session = session; }
  static async load(onnxUrl) {
    const session = await ort.InferenceSession.create(onnxUrl, { executionProviders: ["wasm"] });
    return new WebGrader(session);
  }
  async pose(srcMat) { return runPose(this.session, srcMat); }
  async grade(srcMat) {
    const ordered = orderColumns(await this.pose(srcMat));
    const ncol = ordered.length;
    const results = [], colsMeta = [];
    let qno = 0;
    for (const [colIdx, obj] of ordered) {
      const cls = obj.cls, nr = CFG.COL_ROWS[cls] || 25;
      const { opt, row } = fixedCenters(nr);
      const cimg = cropGrid(srcMat, obj, nr, colIdx === ncol - 1);
      const ans = pixelCountAnswers(cimg, cls, row, opt);
      const colAns = {};
      for (const [rr, a] of ans) { qno++; results.push({ question: qno, col: colIdx, row: rr, cls, answer: a }); colAns[rr] = a; }
      colsMeta.push({ col: colIdx, cls, nr, img: cimg, opt, ans: colAns });
    }
    return { results, colsMeta };
  }
}

// ── Ensemble: โหวตจากคู่ (pose × reader) ที่ดีที่สุด N คู่ ─────────────────────
class EnsembleGrader {
  constructor(poseSessions, bboxSessions) {
    this.pose = poseSessions;     // [{name, session}]
    this.bbox = bboxSessions;     // [{name, conf, session}]
  }

  static async load(onProgress) {
    const open = url => ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
    const pose = [], bbox = [];
    let done = 0; const total = ENSEMBLE.POSE.length + ENSEMBLE.BBOX.length;
    const tick = n => { if (onProgress) onProgress(++done, total, n); };
    for (const name of ENSEMBLE.POSE) { pose.push({ name, session: await open(`models/${name}.onnx`) }); tick(name); }
    for (const b of ENSEMBLE.BBOX) { bbox.push({ ...b, session: await open(`models/${b.name}.onnx`) }); tick(b.name); }
    return new EnsembleGrader(pose, bbox);
  }

  // รัน 1 pose: หา/จัดคอลัมน์ → crop ทุกคอลัมน์ → อ่านด้วย reader ที่ระบุ (หลายตัว)
  // readerSpecs: ["pixel_count" | ชื่อ bbox, ...] ; คืน { maps: Map<q,ans> ต่อ reader, qcount, colsMeta }
  async _gradeOnePose(poseSession, srcMat, readerSpecs, keep, signal = null) {
    await checkAbort(signal);
    const ordered = orderColumns(await runPose(poseSession, srcMat));
    const ncol = ordered.length;
    const maps = readerSpecs.map(() => new Map());
    const colsMeta = [];
    let qno = 0;
    for (const [colIdx, obj] of ordered) {
      const cls = obj.cls, nr = CFG.COL_ROWS[cls] || 25;
      const { opt, row } = fixedCenters(nr);
      const cimg = cropGrid(srcMat, obj, nr, colIdx === ncol - 1);
      const baseQ = qno;
      let pixelAns = null;   // คำนวณครั้งเดียว (ใช้ทั้งเป็น reader และ cellInfo)
      for (let ri = 0; ri < readerSpecs.length; ri++) {
        const spec = readerSpecs[ri];
        let ans;
        if (spec === "pixel_count") {
          if (!pixelAns) pixelAns = pixelCountAnswers(cimg, cls, row, opt);
          ans = pixelAns;
        } else {
          await checkAbort(signal);
          const bm = this.bbox.find(b => b.name === spec);
          const dets = await runDetect(bm.session, cimg, Math.min(bm.conf, ENSEMBLE.DETECT_FLOOR));
          ans = bboxAnswers(dets, cls, row, opt, cimg.cols, bm.conf);
        }
        for (const [rr, a] of ans) maps[ri].set(baseQ + rr, a);
      }
      if (keep) {
        if (!pixelAns) pixelAns = pixelCountAnswers(cimg, cls, row, opt);
        const cellInfo = {};
        for (const [rr, , nf] of pixelAns) cellInfo[rr] = { nFilled: nf };
        colsMeta.push({ col: colIdx, cls, nr, img: cimg, opt, ans: {}, flags: {}, cellInfo });
      } else cimg.delete();
      qno += nr;
    }
    return { maps, qcount: qno, colsMeta };
  }

  async grade(srcMat, level = 12, signal = null) {
    const nPairs = Math.min(Math.max(TIERS[level] || level || 12, 1), PAIRS.length);
    const selected = PAIRS.slice(0, nPairs);
    const quorum = (nPairs >> 1) + 1;             // เสียงข้างมาก
    const primaryPose = selected[0].pose;          // pose ของคู่ที่ดีที่สุด = ภาพ viz
    // จัดกลุ่มตาม pose เพื่อรัน pose+crop ครั้งเดียวต่อ pose
    const byPose = new Map();
    for (const p of selected) {
      if (!byPose.has(p.pose)) byPose.set(p.pose, []);
      byPose.get(p.pose).push(p.reader);
    }
    const allMethods = [];   // Map<q,ans> หนึ่งตัวต่อคู่
    let maxQ = 0, primaryCols = null;
    for (const [poseName, readers] of byPose) {
      await checkAbort(signal);
      const ps = this.pose.find(p => p.name === poseName);
      const keep = poseName === primaryPose;
      const { maps, qcount, colsMeta } = await this._gradeOnePose(ps.session, srcMat, readers, keep, signal);
      allMethods.push(...maps);
      maxQ = Math.max(maxQ, qcount);
      if (keep) primaryCols = colsMeta;
    }
    // โหวตรายข้อ (เก็บอันดับ 1 และ 2 เพื่อตัดสินความ "ชัดเจน")
    const total = allMethods.length;   // = nPairs
    const voted = {};
    for (let q = 1; q <= maxQ; q++) {
      const tally = new Map();
      for (const m of allMethods) if (m.has(q)) { const a = String(m.get(q)); tally.set(a, (tally.get(a) || 0) + 1); }
      let bestA = "None", bestV = 0, secondV = 0;
      for (const [a, v] of tally) {
        if (v > bestV) { secondV = bestV; bestV = v; bestA = a; }
        else if (v > secondV) secondV = v;
      }
      voted[q] = { answer: bestA, votes: bestV, second: secondV, total,
                   confident: bestV > secondV && bestV >= quorum };
    }
    // map คำตอบโหวตกลับเข้า colsMeta (pose ที่ดีที่สุด) + ติดธง uncertain/multi/empty
    const first = colFirstQuestion(primaryCols || []);
    const results = [];
    for (const m of (primaryCols || [])) {
      const fq = first[m.col];
      for (let row = 0; row < m.nr; row++) {
        const q = fq + row, v = voted[q];
        const ans = v ? v.answer : "None";
        const nFilled = m.cellInfo[row + 1] ? m.cellInfo[row + 1].nFilled : null;
        const empty = ans === "None";
        const multi = nFilled != null && nFilled >= 2;
        const confident = v ? v.confident : false;
        m.ans[row + 1] = ans;
        m.flags[row + 1] = { confident, empty, multi };
        results.push({ question: q, col: m.col, row: row + 1, cls: m.cls, answer: ans,
                       votes: v ? v.votes : 0, second: v ? v.second : 0, total,
                       confident, empty, multi });
      }
    }
    results.sort((a, b) => a.question - b.question);
    return { results, colsMeta: primaryCols || [] };
  }
}

// ── format / scoring (เลียน grader_core.py) ──────────────────────────────────
function classSignature(colsMeta) {
  return [...colsMeta].sort((a, b) => a.col - b.col).map(m => m.cls);
}
function signatureText(sig) {
  if (!sig.length) return "0 คอลัมน์";
  return `${sig.length} คอลัมน์ (จำนวนข้อ: ${sig.map(c => CLASS_LABEL[c]).join(", ")})`;
}
function compareSignatures(keySig, sheetSig) {
  if (keySig === null) return [true, "เฉลยมาจาก CSV — ไม่มีข้อมูลรูปแบบกระดาษให้เทียบ"];
  if (JSON.stringify(keySig) === JSON.stringify(sheetSig)) return [true, `รูปแบบกระดาษตรงกับเฉลย: ${signatureText(sheetSig)}`];
  const total = sig => sig.reduce((s, c) => s + (CLASS_LABEL[c] || 0), 0);
  const kt = total(keySig), st = total(sheetSig);
  let head;
  if (kt !== st) {
    head = `รูปภาพไม่สมบูรณ์ หรือชนิดของกระดาษคำตอบไม่ตรงกับเฉลย — ระบบตรวจสอบว่าเฉลยเป็นกระดาษคำตอบแบบ ${kt} ข้อ แต่กระดาษคำตอบเป็นแบบ ${st} ข้อ โปรดตรวจสอบ`;
  } else if (keySig.length !== sheetSig.length) {
    head = `จำนวนคอลัมน์ไม่ตรง (เฉลย ${keySig.length} ≠ ใบคำตอบ ${sheetSig.length}) — รูปภาพอาจไม่สมบูรณ์ โปรดตรวจสอบ`;
  } else {
    head = `รูปแบบคอลัมน์ของกระดาษไม่ตรงกับเฉลย (จำนวนข้อรวมเท่ากัน ${kt} ข้อ แต่การจัดคอลัมน์ต่างกัน) โปรดตรวจสอบ`;
  }
  return [false, `${head}\n  เฉลย    : ${signatureText(keySig)}\n  ใบคำตอบ : ${signatureText(sheetSig)}`];
}
function colFirstQuestion(colsMeta) {
  const first = {}; let q = 1;
  for (const m of [...colsMeta].sort((a, b) => a.col - b.col)) { first[m.col] = q; q += m.nr; }
  return first;
}
function resultsToKey(results) { const k = {}; for (const r of results) k[r.question] = r.answer; return k; }
function scoreResults(results, key) {
  let correct = 0, graded = 0; const wrong = [];
  for (const r of results) {
    if (!(r.question in key)) continue;
    graded++;
    if (String(r.answer) === String(key[r.question])) correct++;
    else wrong.push([r.question, r.answer, key[r.question]]);
  }
  return { correct, graded, wrong };
}
function formatKeyByColumn(colsMeta) {
  const first = colFirstQuestion(colsMeta); const lines = [];
  for (const m of [...colsMeta].sort((a, b) => a.col - b.col)) {
    const fq = first[m.col];
    const seq = [];
    for (let row = 0; row < m.nr; row++) seq.push(`${fq + row}:${m.ans[row + 1] || "None"}`);
    lines.push(`คอลัมน์ ${m.col + 1} (Class${CLASS_LABEL[m.cls]} • ${m.nr} ข้อ): ${seq.join(" ")}`);
  }
  return lines.join("\n");
}

// ── annotate: เลขข้อซ้าย + วงกลมคำตอบ (น้ำเงิน/เขียว/แดง) + แถบขวาถูก/ผิด ──────
function drawCheck(mat, cx, cy, color) {   // เครื่องหมายถูก ✓
  cv.line(mat, new cv.Point(cx - 9, cy + 1), new cv.Point(cx - 2, cy + 8), color, 3);
  cv.line(mat, new cv.Point(cx - 2, cy + 8), new cv.Point(cx + 10, cy - 9), color, 3);
}
function drawCross(mat, cx, cy, color) {   // เครื่องหมายผิด ✗
  cv.line(mat, new cv.Point(cx - 8, cy - 8), new cv.Point(cx + 8, cy + 8), color, 3);
  cv.line(mat, new cv.Point(cx - 8, cy + 8), new cv.Point(cx + 8, cy - 8), color, 3);
}
function annotate(colsMeta, key = null, colFirstQ = null) {
  if (!colsMeta.length) return null;
  const BLUE = new cv.Scalar(255, 0, 0, 255);        // คำตอบ (โหมดเฉลย/ยังไม่ตรวจ)
  const GREEN = new cv.Scalar(0, 160, 0, 255);       // ถูก
  const RED = new cv.Scalar(0, 0, 255, 255);         // ผิด
  const ORANGE = new cv.Scalar(0, 165, 255, 255);    // ไม่มั่นใจ
  const MAGENTA = new cv.Scalar(255, 0, 255, 255);   // มากกว่า 1 คำตอบ
  const PURPLE = new cv.Scalar(211, 0, 148, 255);    // ม่วง — กริดข้อที่ไม่มั่นใจ (มากกว่า 1 คำตอบ)
  const YELLOW = new cv.Scalar(0, 255, 255, 255);    // เหลือง — กริดข้อที่ไม่ชัดเจน (โหวตไม่ชัด)
  const BLACK = new cv.Scalar(40, 40, 40, 255);
  const grading = key !== null;
  const numFirst = colFirstQuestion(colsMeta);       // เลขข้อต่อเนื่องตามคลาส
  const LEFTPAD = 60, RIGHTPAD = grading ? 64 : 8, W = CFG.N_CHOICES * CFG.CELL;
  const crops = [];
  for (const m of colsMeta) {
    const h = m.img.rows, tw = LEFTPAD + W + RIGHTPAD;
    const c = new cv.Mat(h, tw, m.img.type(), new cv.Scalar(255, 255, 255, 255));
    const roi = c.roi(new cv.Rect(LEFTPAD, 0, W, h));
    m.img.copyTo(roi); roi.delete();
    // ตีกริดสีเขียวอ่อนทับช่องคำตอบ (เส้นหนา)
    const GRID = new cv.Scalar(120, 210, 120, 255);
    for (let i = 0; i <= CFG.N_CHOICES; i++) {
      const gx = LEFTPAD + i * CFG.CELL;
      cv.line(c, new cv.Point(gx, 0), new cv.Point(gx, m.nr * CFG.CELL), GRID, 2);
    }
    for (let r = 0; r <= m.nr; r++) {
      const gy = r * CFG.CELL;
      cv.line(c, new cv.Point(LEFTPAD, gy), new cv.Point(LEFTPAD + W, gy), GRID, 2);
    }
    const fq = numFirst[m.col];
    for (let row = 0; row < m.nr; row++) {
      const gq = fq + row;
      const a = m.ans[row + 1] || "None";
      const fl = (m.flags && m.flags[row + 1]) || {};
      const y = Math.floor((row + 0.5) * CFG.CELL);
      // กริดเตือน: ม่วง = มากกว่า 1 คำตอบ (ไม่มั่นใจ), เหลือง = โหวตไม่ชัด (ไม่ชัดเจน)
      if (fl.multi) {
        cv.rectangle(c, new cv.Point(LEFTPAD, row * CFG.CELL),
          new cv.Point(LEFTPAD + W, (row + 1) * CFG.CELL), PURPLE, 3);
      } else if (fl.confident === false) {
        cv.rectangle(c, new cv.Point(LEFTPAD, row * CFG.CELL),
          new cv.Point(LEFTPAD + W, (row + 1) * CFG.CELL), YELLOW, 3);
      }
      // เลขข้อ (ซ้าย)
      cv.putText(c, String(gq), new cv.Point(6, y + 7), cv.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2);
      // สีคำตอบ
      let correct = null, color = BLUE;
      if (grading && gq in key) { correct = String(a) === String(key[gq]); color = correct ? GREEN : RED; }
      if (CFG.OPT_NAMES.includes(a)) {
        const x = LEFTPAD + Math.floor(m.opt[a]);
        cv.circle(c, new cv.Point(x, y), 26, color, 2);
        cv.putText(c, a, new cv.Point(x - 10, y + 8), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        if (!fl.confident) cv.putText(c, "?", new cv.Point(x + 24, y - 14), cv.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2);
      } else {
        cv.putText(c, "-", new cv.Point(LEFTPAD + 4, y + 8), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(160, 160, 160, 255), 2);
      }
      // โหมดตรวจ: ถ้าผิด วงเขียวบอกช่องที่ถูก + แถบขวา ✓/✗
      if (grading && correct !== null) {
        if (!correct && CFG.OPT_NAMES.includes(String(key[gq]))) {
          const kx = LEFTPAD + Math.floor(m.opt[String(key[gq])]);
          cv.circle(c, new cv.Point(kx, y), 26, GREEN, 2);
        }
        const sx = LEFTPAD + W + RIGHTPAD / 2;
        if (correct) drawCheck(c, sx, y, GREEN); else drawCross(c, sx, y, RED);
      }
    }
    crops.push(c);
  }
  const h = Math.max(...crops.map(c => c.rows));
  const padded = crops.map(c => {
    const p = new cv.Mat();
    cv.copyMakeBorder(c, p, 0, h - c.rows, 6, 6, cv.BORDER_CONSTANT, new cv.Scalar(255, 255, 255, 255));
    c.delete();
    return p;
  });
  const vv = new cv.MatVector(); padded.forEach(p => vv.push_back(p));
  const canvas = new cv.Mat();
  cv.hconcat(vv, canvas);
  vv.delete(); padded.forEach(p => p.delete());
  return canvas;
}

window.WebPipeline = {
  CFG, CLASS_LABEL, ENSEMBLE, TIERS, PAIRS, tierQuorum, WebGrader, EnsembleGrader,
  classSignature, signatureText, compareSignatures, colFirstQuestion,
  resultsToKey, scoreResults, formatKeyByColumn, annotate,
};

/* app.js — UI glue: โหลดโมเดล, เฉลย/ตรวจคำตอบ, อินพุตไฟล์/กล้อง, แสดงผล
 * อาศัย window.WebPipeline (pipeline.js), global cv (OpenCV.js), global ort (onnxruntime-web)
 */
const P = window.WebPipeline;
let cvReady = false, grader = null, key = null, keySig = null;
let batchResults = [];   // ผลเต็มต่อไฟล์ (สำหรับ popup รายใบ + ชื่อจริง/ชื่อเล่น)
let level = 3;   // ระดับการโหวต (จำนวนคู่ pose×reader ที่ดีที่สุด: 12/9/6/3/1) — ค่าเริ่มต้น 3
const EXPECTED_COLS = 4;   // กระดาษคำตอบมาตรฐาน = 4 คอลัมน์ ; ไม่ครบ/เกิน = detect ผิดพลาด
const DETECT_HINT = "การ detect ผิดพลาด — กรุณาปรับภาพ (ถ่ายให้ตรง แสงสว่างพอ เห็นกระดาษครบทั้งใบ ไม่เอียง/เงา) แล้วลองใหม่";

// ไฟล์ที่เลือกไว้ (รอกดปุ่มประมวลผล)
let selKeyFile = null, selGradeFile = null, selBatchFiles = null, gradeMode = "single";

const TIER_DESC = {
  12: "12 คู่ — แม่นยำสุด แต่ช้า",
  9:  "9 คู่ — แม่นยำสูง",
  6:  "6 คู่ — สมดุล",
  3:  "3 คู่ — เร็ว สมดุล (ค่าเริ่มต้น)",
  1:  "1 คู่ — เร็วสุด แต่อาจผิดพลาด",
};
const TIER_COLOR = {
  12: "var(--ok)",    // แม่นยำสุด
  9:  "var(--ok)",    // แม่นยำสูง
  6:  "var(--warn)",  // สมดุล
  3:  "var(--warn)",  // เร็ว สมดุล
  1:  "var(--bad)",   // เร็วสุด แต่อาจผิดพลาด
};
function tierQuorum(l) { return P.tierQuorum(l); }
function setTier(l) {
  level = l;
  document.querySelectorAll("#tierBtns .tier").forEach(b =>
    b.classList.toggle("active", +b.dataset.level === l));
  $("tierInfo").innerHTML =
    `<span style="color:${TIER_COLOR[l]};font-weight:600">${TIER_DESC[l]}</span>` +
    ` · expand-crop · โหวต ≥${tierQuorum(l)}/${l}`;
}

const $ = id => document.getElementById(id);
function setStatus(msg, cls) { const s = $("status"); s.textContent = msg; s.style.color = noteColor(cls); }
function noteColor(cls) { return cls === "bad" ? "var(--bad)" : cls === "warn" ? "var(--warn)" : "var(--ok)"; }
function setNote(id, msg, cls) { const e = $(id); e.textContent = msg; e.style.color = noteColor(cls); }

document.querySelectorAll("#tierBtns .tier").forEach(b =>
  b.onclick = () => setTier(+b.dataset.level));
setTier(3);

// ── โหลดไลบรารี (onnxruntime-web + OpenCV.js) แบบ lazy ครั้งแรกตอนใช้งาน ──
function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = src; s.async = true;
    s.onload = () => res();
    s.onerror = () => rej(new Error("โหลดไลบรารีไม่สำเร็จ: " + src));
    document.head.appendChild(s);
  });
}
let libsPromise = null;
async function loadLibs() {
  if (cvReady) return;
  if (!libsPromise) {
    libsPromise = (async () => {
      setStatus("กำลังโหลดไลบรารี…", "warn");
      await loadScript("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
      await loadScript("https://docs.opencv.org/4.x/opencv.js");
      await new Promise(res => {
        if (window.cv && cv.Mat) return res();
        cv.onRuntimeInitialized = () => res();
      });
      cvReady = true;
    })();
  }
  return libsPromise;
}

// ── โหลด ensemble (pose + bbox) แบบ lazy: โหลดไลบรารี+โมเดลครั้งแรกตอนกดประมวลผล ──
let modelPromise = null;
async function ensureModel() {
  if (grader) return grader;
  if (!modelPromise) {
    modelPromise = (async () => {
      try {
        await loadLibs();
        setStatus("กำลังโหลดโมเดล ensemble…", "warn");
        grader = await P.EnsembleGrader.load((done, total, name) =>
          setStatus(`กำลังโหลดโมเดล ${done}/${total} (${name})…`, "warn"));
        setStatus(`พร้อมใช้งาน · ${level} คู่ (โหวต ≥${tierQuorum(level)}/${level})`);
        return grader;
      } catch (e) {
        modelPromise = null;   // ให้ลองใหม่ได้
        setStatus("โหลดไม่สำเร็จ: " + e, "bad"); console.error(e);
        return null;
      }
    })();
  }
  return modelPromise;
}
setStatus("พร้อม — เลือกไฟล์แล้วกด 'ประมวลผลเฉลย' (โหลดครั้งแรกอัตโนมัติ)");

// ── ตัวช่วยแปลงภาพ ──
function fileToImage(file) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => res(img);
    img.onerror = rej;
    img.src = URL.createObjectURL(file);
  });
}
function imgElToBGR(imgEl) {
  const c = document.createElement("canvas");
  c.width = imgEl.naturalWidth || imgEl.videoWidth || imgEl.width;
  c.height = imgEl.naturalHeight || imgEl.videoHeight || imgEl.height;
  c.getContext("2d").drawImage(imgEl, 0, 0, c.width, c.height);
  const rgba = cv.imread(c);
  const bgr = new cv.Mat();
  cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR);
  rgba.delete();
  return bgr;
}
async function fileToBGR(file) { return imgElToBGR(await fileToImage(file)); }
// center-crop ภาพจากกล้องให้เป็นสัดส่วน 4:3 (กว้าง:สูง) + ดิจิทัลซูม → คืน canvas
function cropCanvas43(el) {
  let w = el.naturalWidth || el.videoWidth || el.width;
  let h = el.naturalHeight || el.videoHeight || el.height;
  let ox = 0, oy = 0;
  if (digitalZoom && camZoom > 1) {     // ดิจิทัลซูม: ตัดบริเวณกลางเข้าไป
    const zw = w / camZoom, zh = h / camZoom;
    ox = (w - zw) / 2; oy = (h - zh) / 2; w = zw; h = zh;
  }
  let cw = w, ch = Math.round(w * 3 / 4);
  if (ch > h) { ch = h; cw = Math.round(h * 4 / 3); }
  const sx = Math.round(ox + (w - cw) / 2), sy = Math.round(oy + (h - ch) / 2);
  cw = Math.round(cw); ch = Math.round(ch);
  const c = document.createElement("canvas");
  c.width = cw; c.height = ch;
  c.getContext("2d").drawImage(el, sx, sy, cw, ch, 0, 0, cw, ch);
  return c;
}
function imgElToBGR43(el) {
  const c = cropCanvas43(el);
  const rgba = cv.imread(c);
  const bgr = new cv.Mat();
  cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR);
  rgba.delete();
  return bgr;
}
function dataURLToImage(url) {
  return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = url; });
}
async function dataURLToBGR(url) { return imgElToBGR(await dataURLToImage(url)); }
function showBGR(canvasId, bgrMat) {
  const rgba = new cv.Mat();
  cv.cvtColor(bgrMat, rgba, cv.COLOR_BGR2RGBA);
  cv.imshow(canvasId, rgba);
  rgba.delete();
}
function bgrToDataURL(bgrMat) {
  const rgba = new cv.Mat();
  cv.cvtColor(bgrMat, rgba, cv.COLOR_BGR2RGBA);
  const c = document.createElement("canvas");
  cv.imshow(c, rgba);
  rgba.delete();
  return c.toDataURL("image/png");
}

// ── ป้ายแจ้งธงรายข้อ (uncertain/multi/empty) ──
function renderFlags(elId, results) {
  const el = $(elId); el.innerHTML = "";
  const unc = results.filter(r => r.confident === false).length;
  const multi = results.filter(r => r.multi).length;
  const empty = results.filter(r => r.empty).length;
  const add = (cls, txt) => { const s = document.createElement("span"); s.className = "flag " + cls; s.textContent = txt; el.appendChild(s); };
  if (unc) add("unc", `ไม่ชัดเจน ${unc} ข้อ`);
  if (multi) add("multi", `มีความไม่มั่นใจ ${multi} ข้อ โปรดตรวจสอบ`);
  if (empty) add("empty", `ช่องว่าง ${empty} ข้อ`);
  if (!unc && !multi && !empty) add("good", "ทุกข้อชัดเจน");
}
// คำตอบเฉลย — แสดงเป็นคอลัมน์ตามกระดาษจริง (4 คอลัมน์) แต่ละช่องเรียงข้อแนวตั้ง
function keyHTML(colsMeta) {
  const first = P.colFirstQuestion(colsMeta);
  const cols = [...colsMeta].sort((a, b) => a.col - b.col).map(m => {
    const fq = first[m.col];
    let rows = "";
    for (let r = 0; r < m.nr; r++) rows += `<div>ข้อ ${fq + r}: <b class="ok">${esc(m.ans[r + 1] || "-")}</b></div>`;
    return `<div style="flex:1 1 0;min-width:90px">` +
      `<div style="color:var(--muted);border-bottom:1px solid var(--line);margin-bottom:4px;padding-bottom:3px">คอลัมน์ ${m.col + 1} · ${m.nr} ข้อ</div>` +
      rows + `</div>`;
  }).join("");
  return `<div style="display:flex;gap:14px">${cols}</div>`;
}
// รายการข้อที่ตอบผิด — รูปแบบ "ข้อ x. ตอบ C คำตอบที่ถูกต้องคือ B" จัด 4 คอลัมน์
function wrongHTML(wrong, detail) {
  let html;
  if (!wrong.length) {
    html = `<span class="ok">ถูกทุกข้อ ✓</span>`;
  } else {
    const items = wrong.map(([q, g, ex]) =>
      `<div style="break-inside:avoid;margin-bottom:5px">ข้อ ${q}. ตอบ <b class="bad">${esc(g)}</b> คำตอบที่ถูกต้องคือ <b class="ok">${esc(ex)}</b></div>`
    ).join("");
    html = `<div style="column-count:4;column-gap:16px">${items}</div>`;
  }
  if (detail) html += `<div style="margin-top:12px;color:var(--muted)">— ข้อควรตรวจสอบ —<br>${esc(detail).replace(/\n/g, "<br>")}</div>`;
  return html;
}
function flagDetailText(results) {
  const lines = [];
  const unc = results.filter(r => r.confident === false);
  const multi = results.filter(r => r.multi);
  const empty = results.filter(r => r.empty);
  if (unc.length) lines.push("ไม่ชัดเจน: " + unc.map(r => `ข้อ ${r.question}(${r.votes}/${r.total})`).join(" "));
  if (multi.length) lines.push("มีความไม่มั่นใจ (โปรดตรวจสอบ): " + multi.map(r => `ข้อ ${r.question}`).join(" "));
  if (empty.length) lines.push("ช่องว่าง: " + empty.map(r => `ข้อ ${r.question}`).join(" "));
  return lines.join("\n");
}

// ── กล้อง ──
let camStream = null, camResolve = null;
async function camPermissionState() {
  try {
    if (navigator.permissions && navigator.permissions.query) {
      const st = await navigator.permissions.query({ name: "camera" });
      return st.state;   // "granted" | "denied" | "prompt"
    }
  } catch (_) {}
  return "unknown";
}
async function camErrorMsg(e) {
  switch (e && e.name) {
    case "NotAllowedError": case "SecurityError": {
      // ถ้าสิทธิ์ถูกบล็อกถาวร เบราว์เซอร์จะไม่ถามอีก ต้องรีเซ็ตเอง
      if (await camPermissionState() === "denied")
        return "กล้องถูกบล็อกไว้ เบราว์เซอร์จะไม่ถามสิทธิ์อีก — โปรดคลิกไอคอน 🔒/กล้อง ข้างแถบที่อยู่ → 'กล้อง' → เลือก 'อนุญาต' แล้วรีเฟรชหน้า (หรือ Chrome: Settings → ความเป็นส่วนตัว → การตั้งค่าไซต์ → กล้อง) หรือใช้ปุ่ม 'เลือกไฟล์' แทน";
      return "คุณยังไม่ได้กดอนุญาต — เมื่อเบราว์เซอร์ถามสิทธิ์กล้อง โปรดกด 'อนุญาต' แล้วลองใหม่ หรือใช้ปุ่ม 'เลือกไฟล์' แทน";
    }
    case "NotFoundError": case "DevicesNotFoundError": case "OverconstrainedError":
      return "ไม่พบกล้องในเครื่อง — โปรดต่อกล้อง หรือใช้ปุ่ม 'เลือกไฟล์' แทน";
    case "NotReadableError": case "TrackStartError":
      return "กล้องถูกใช้งานโดยโปรแกรมอื่นอยู่ — ปิดโปรแกรมที่ใช้กล้องแล้วลองใหม่";
    default:
      return "เปิดกล้องไม่ได้: " + ((e && e.message) || e);
  }
}
// ทิศกล้องปัจจุบัน: "environment" = หลัง, "user" = หน้า
let camFacing = "environment";
// ── ซูมกล้อง: ฮาร์ดแวร์ก่อน (Android Chrome) ไม่งั้น digital crop ──
let camZoom = 1, digitalZoom = false;
const ZOOM_MAX = 4;
function clampZoom(z, max) { return Math.min(max, Math.max(1, z)); }
async function applyZoom(stream, videoEl, z) {
  const track = stream && stream.getVideoTracks && stream.getVideoTracks()[0];
  const caps = track && track.getCapabilities ? track.getCapabilities() : null;
  if (caps && caps.zoom) {                              // ฮาร์ดแวร์ซูม
    const maxLogical = caps.zoom.min > 0 ? caps.zoom.max / caps.zoom.min : caps.zoom.max;
    camZoom = clampZoom(z, Math.max(1.01, maxLogical));
    digitalZoom = false;
    const val = Math.min(caps.zoom.max, Math.max(caps.zoom.min, caps.zoom.min * camZoom));
    try { await track.applyConstraints({ advanced: [{ zoom: val }] }); } catch (_) {}
    if (videoEl) videoEl.style.transform = "";
  } else {                                              // digital ซูม (crop + scale พรีวิว)
    camZoom = clampZoom(z, ZOOM_MAX);
    digitalZoom = camZoom > 1;
    if (videoEl) { videoEl.style.transformOrigin = "center"; videoEl.style.transform = `scale(${camZoom})`; }
  }
}
function resetZoom(videoEl) { camZoom = 1; digitalZoom = false; if (videoEl) videoEl.style.transform = ""; }
function attachPinchZoom(videoEl, getStream) {
  const pts = new Map();
  let startDist = 0, startZoom = 1;
  const dist = () => { const [a, b] = [...pts.values()]; return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY); };
  videoEl.style.touchAction = "none";
  videoEl.addEventListener("pointerdown", e => {
    pts.set(e.pointerId, e);
    if (pts.size === 2) { startDist = dist(); startZoom = camZoom; }
  });
  videoEl.addEventListener("pointermove", e => {
    if (!pts.has(e.pointerId)) return;
    pts.set(e.pointerId, e);
    if (pts.size === 2 && startDist > 0) applyZoom(getStream(), videoEl, startZoom * dist() / startDist);
  });
  const up = e => { pts.delete(e.pointerId); if (pts.size < 2) startDist = 0; };
  videoEl.addEventListener("pointerup", up);
  videoEl.addEventListener("pointercancel", up);
}
// ขอกล้องตามทิศที่ระบุ บังคับ exact ก่อน ถ้าเครื่องไม่มีค่อย fallback
async function getCamStream(facing = camFacing) {
  const size = { width: { ideal: 1280 }, height: { ideal: 960 }, aspectRatio: { ideal: 4 / 3 } };
  try {
    return await navigator.mediaDevices.getUserMedia({ video: { ...size, facingMode: { exact: facing } } });
  } catch (e) {
    if (e && (e.name === "OverconstrainedError" || e.name === "NotFoundError" || e.name === "ConstraintNotSatisfiedError")) {
      return await navigator.mediaDevices.getUserMedia({ video: { ...size, facingMode: facing } });
    }
    throw e;
  }
}
function openCamera() {
  return new Promise(async (resolve) => {
    camResolve = resolve;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("เบราว์เซอร์นี้ใช้กล้องไม่ได้ — ต้องเปิดผ่าน https หรือ localhost (ใช้ปุ่ม 'เลือกไฟล์' แทนได้)", "bad");
      resolve(null); camResolve = null; return;
    }
    if (await camPermissionState() === "denied") {
      setStatus(await camErrorMsg({ name: "NotAllowedError" }), "bad");
      resolve(null); camResolve = null; return;
    }
    setStatus("กำลังขอสิทธิ์ใช้กล้อง… โปรดกด 'อนุญาต' ในเบราว์เซอร์", "warn");
    try {
      camStream = await getCamStream();
      $("cam").srcObject = camStream;
      resetZoom($("cam"));
      $("camModal").style.display = "flex";
      setStatus("พร้อมใช้งาน");
    } catch (e) { setStatus(await camErrorMsg(e), "bad"); resolve(null); camResolve = null; }
  });
}
function closeCamera(result) {
  $("camModal").style.display = "none";
  if (camStream) { camStream.getTracks().forEach(t => t.stop()); camStream = null; }
  if (camResolve) { camResolve(result); camResolve = null; }
}
$("camShot").onclick = () => {
  closeCamera(imgElToBGR43($("cam")));   // จับภาพแบบ crop 4:3
};
$("camCancel").onclick = () => closeCamera(null);
$("camFlip").onclick = async () => {
  const next = camFacing === "environment" ? "user" : "environment";
  // มือถือเปิดกล้องได้ทีละตัว — ต้องปิดตัวเดิมก่อนค่อยขอทิศใหม่
  if (camStream) { camStream.getTracks().forEach(t => t.stop()); camStream = null; $("cam").srcObject = null; }
  let s;
  try { s = await getCamStream(next); }
  catch (e) {
    setStatus(await camErrorMsg(e), "bad");
    try { camStream = await getCamStream(camFacing); $("cam").srcObject = camStream; } catch (_) {}
    return;
  }
  camFacing = next; camStream = s; $("cam").srcObject = s; resetZoom($("cam"));
};
$("camModal").onclick = e => { if (e.target === $("camModal")) closeCamera(null); };
document.addEventListener("keydown", e => { if (e.key === "Escape" && $("camModal").style.display === "flex") closeCamera(null); });

// ── ถ่ายหลายใบจากกล้อง (batch capture) — ถ่าย → กรอกชื่อตอนพรีวิว → ถัดไป → เสร็จก็ตรวจทั้งชุด ──
let capStream = null, capItems = [], capPendingURL = null, capEditIdx = -1;
function capRenumber() { capItems.forEach((it, i) => it.name = `ถ่าย_${i + 1}`); }
function capRenderThumbs() {
  const box = $("capThumbs"); box.innerHTML = "";
  capItems.forEach((it, i) => {
    const nm = [it.realName, it.nick && `(${it.nick})`].filter(Boolean).join(" ") || it.name;
    const d = document.createElement("div");
    d.style.cssText = "position:relative;width:84px;cursor:pointer";
    d.innerHTML =
      `<img src="${it.dataURL}" alt="" style="width:84px;height:63px;object-fit:cover;border:1px solid var(--line);border-radius:6px;display:block">` +
      `<div style="font-size:11px;color:var(--muted);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(nm)}</div>` +
      `<button title="ลบ" style="position:absolute;top:-6px;right:-6px;width:20px;height:20px;border-radius:50%;border:none;background:var(--bad);color:#fff;cursor:pointer;line-height:1;padding:0">✕</button>`;
    d.querySelector("img").onclick = () => capEdit(i);
    d.querySelector("div").onclick = () => capEdit(i);
    d.querySelector("button").onclick = e => { e.stopPropagation(); capItems.splice(i, 1); capRenumber(); capRenderThumbs(); };
    box.appendChild(d);
  });
  $("capCount").textContent = capItems.length + " ใบ";
  $("capDone").disabled = !capItems.length;
}
function capShowLive() { $("capLive").style.display = ""; $("capPreview").style.display = "none"; capPendingURL = null; }
function capShowPreview(url, real, nick) {
  capPendingURL = url; $("capPrevImg").src = url;
  $("capReal").value = real || ""; $("capNick").value = nick || "";
  $("capLive").style.display = "none"; $("capPreview").style.display = "";
}
function capEdit(i) { const it = capItems[i]; if (!it) return; capEditIdx = i; capShowPreview(it.dataURL, it.realName, it.nick); }
async function openCapModal() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setNote("gradeStatus", "เบราว์เซอร์นี้ใช้กล้องไม่ได้ — ต้องเปิดผ่าน https หรือ localhost", "bad"); return;
  }
  capItems = []; capEditIdx = -1; capPendingURL = null;
  capRenderThumbs(); capShowLive();
  setStatus("กำลังขอสิทธิ์ใช้กล้อง…", "warn");
  try {
    capStream = await getCamStream();
    $("capCam").srcObject = capStream; resetZoom($("capCam"));
    $("capModal").style.display = "flex";
    setStatus("ถ่ายทีละใบ กรอกชื่อ แล้วกด 'เสร็จ' เพื่อตรวจทั้งชุด");
  } catch (e) { setStatus(await camErrorMsg(e), "bad"); }
}
function closeCapModal() {
  $("capModal").style.display = "none";
  if (capStream) { capStream.getTracks().forEach(t => t.stop()); capStream = null; }
  $("capCam").srcObject = null;
}
$("capShot").onclick = () => {
  const url = cropCanvas43($("capCam")).toDataURL("image/jpeg", 0.9);
  if (capEditIdx >= 0) capShowPreview(url, $("capReal").value, $("capNick").value);
  else capShowPreview(url, "", "");
};
$("capRetake").onclick = () => { $("capLive").style.display = ""; $("capPreview").style.display = "none"; };
$("capNext").onclick = () => {
  if (!capPendingURL) return;
  const real = $("capReal").value.trim(), nick = $("capNick").value.trim();
  if (capEditIdx >= 0) { Object.assign(capItems[capEditIdx], { dataURL: capPendingURL, realName: real, nick }); capEditIdx = -1; }
  else capItems.push({ name: "", realName: real, nick, dataURL: capPendingURL });
  capRenumber(); capRenderThumbs(); capShowLive();
};
$("capFlip").onclick = async () => {
  const next = camFacing === "environment" ? "user" : "environment";
  if (capStream) { capStream.getTracks().forEach(t => t.stop()); capStream = null; $("capCam").srcObject = null; }
  let s;
  try { s = await getCamStream(next); }
  catch (e) { setStatus(await camErrorMsg(e), "bad"); try { capStream = await getCamStream(camFacing); $("capCam").srcObject = capStream; } catch (_) {} return; }
  camFacing = next; capStream = s; $("capCam").srcObject = s; resetZoom($("capCam"));
};
$("capCancel").onclick = () => closeCapModal();
$("capDone").onclick = async () => {
  if (!capItems.length) return;
  const items = capItems.map(it => ({ ...it }));
  closeCapModal();
  if (!(await ensureModel())) return;
  $("gradeFileInfo").textContent = `ถ่ายจากกล้อง (ชุด): ${items.length} ใบ`;
  showBatch();
  await withBusy("gradeProcessBtn", () => gradeBatch(items));
};
$("capCamBtn").onclick = () => {
  if (!requireKey("gradeStatus")) return;
  stopRealtime();
  openCapModal();
};
$("capModal").onclick = e => { if (e.target === $("capModal")) closeCapModal(); };

function requireReady() { return true; }   // ไลบรารี/โมเดลโหลดเองตอนกดประมวลผล (ensureModel)

// แสดงสปินเนอร์ระหว่างประมวลผล: ปุ่มที่กดจะหมุน + สปินเนอร์เล็กข้างสถานะ + ปุ่มหยุด
let abortCtrl = null;
function abortSignal() { return abortCtrl ? abortCtrl.signal : null; }
async function withBusy(btnId, fn) {
  const btn = btnId ? $(btnId) : null;
  abortCtrl = new AbortController();
  if (btn) btn.classList.add("busy");
  $("spin").style.display = "inline-block";
  $("stopBtn").style.display = "inline-block";
  try { return await fn(); }
  finally {
    if (btn) btn.classList.remove("busy");
    $("spin").style.display = "none";
    $("stopBtn").style.display = "none";
    abortCtrl = null;
  }
}
$("stopBtn").onclick = () => { if (abortCtrl) { abortCtrl.abort(); setStatus("กำลังยกเลิก…", "warn"); } };

// ── เฉลย: เลือกไฟล์ → รอกดปุ่มประมวลผล ──
$("keyFile").onchange = e => {
  if (!e.target.files[0]) return;
  selKeyFile = e.target.files[0];
  $("keyFileInfo").textContent = "เลือกไฟล์: " + selKeyFile.name;
  setNote("keyStatus", "กดปุ่ม 'ประมวลผลเฉลย' เพื่อหาคำตอบ", "warn");
  $("keyProcessBtn").disabled = false;
  e.target.value = "";
};
$("keyProcessBtn").onclick = async () => {
  if (!requireReady("keyStatus") || !selKeyFile) return;
  if (!(await ensureModel())) return;
  await withBusy("keyProcessBtn", async () => processKey(await fileToBGR(selKeyFile)));
};
$("keyCamBtn").onclick = async () => {
  if (!requireReady("keyStatus")) return;
  if (!(await ensureModel())) return;
  const bgr = await openCamera();
  if (bgr) await withBusy(null, () => processKey(bgr));
};
async function processKey(bgr) {
  setNote("keyStatus", "กำลังประมวลผลเฉลย…", "warn"); setStatus("กำลังประมวลผลเฉลย…", "warn");
  try {
    const { results, colsMeta } = await grader.grade(bgr, level, abortSignal());
    if (!results.length) { setNote("keyStatus", "ตรวจไม่พบคอลัมน์ในใบเฉลย — " + DETECT_HINT, "bad"); bgr.delete(); return; }
    if (colsMeta.length !== EXPECTED_COLS) {
      setNote("keyStatus", `detect ผิดพลาด: พบ ${colsMeta.length}/${EXPECTED_COLS} คอลัมน์ — ${DETECT_HINT}`, "bad");
      $("keyText").textContent = "—"; $("keyFlags").innerHTML = "";
      colsMeta.forEach(m => m.img.delete()); bgr.delete(); return;
    }
    key = P.resultsToKey(results);
    keySig = P.classSignature(colsMeta);
    const canvas = P.annotate(colsMeta);
    showBGR("keyCanvas", canvas); canvas.delete();
    renderFlags("keyFlags", results);
    const detail = flagDetailText(results);
    $("keyText").innerHTML = esc("โครงสร้าง: " + P.signatureText(keySig) + (detail ? "\n" + detail : "")) +
      "\n\n" + keyHTML(colsMeta);
    const unc = results.filter(r => r.confident === false).length;
    setNote("keyStatus", `อ่านเฉลยเสร็จ: ${Object.keys(key).length} ข้อ • ${P.signatureText(keySig)}` + (unc ? ` • ไม่ชัดเจน ${unc}` : ""), unc ? "warn" : "ok");
    setStatus("อ่านเฉลยเสร็จ");
    if (selGradeFile || selBatchFiles) $("gradeProcessBtn").disabled = false;
    colsMeta.forEach(m => m.img.delete());
  } catch (e) {
    if (e.name === "AbortError") { setNote("keyStatus", "ยกเลิกการประมวลผลเฉลยแล้ว", "warn"); setStatus("ยกเลิกแล้ว", "warn"); }
    else { setNote("keyStatus", "อ่านเฉลยล้มเหลว: " + e, "bad"); console.error(e); }
  }
  bgr.delete();
}

// ── ตรวจคำตอบ ──
function requireKey(noteId) {
  if (!key) { setNote(noteId, "กรุณาตั้งเฉลยทางซ้ายก่อน", "bad"); return false; }
  return true;
}
function showSingle() { $("singleView").style.display = "block"; $("batchView").style.display = "none"; $("realtimeView").style.display = "none"; }
function showBatch() { $("singleView").style.display = "none"; $("batchView").style.display = "block"; $("realtimeView").style.display = "none"; }
function showRealtime() { $("singleView").style.display = "none"; $("batchView").style.display = "none"; $("realtimeView").style.display = "block"; }

$("gradeFile").onchange = e => {
  if (!e.target.files[0]) return;
  stopRealtime(); showSingle();
  selGradeFile = e.target.files[0]; selBatchFiles = null; gradeMode = "single";
  $("gradeFileInfo").textContent = "เลือกไฟล์ (เดี่ยว): " + selGradeFile.name;
  setNote("gradeStatus", "กดปุ่ม 'ตรวจคำตอบ' เพื่อตรวจ", "warn");
  $("gradeProcessBtn").disabled = false;
  e.target.value = "";
};
$("batchFiles").onchange = e => {
  const files = [...e.target.files];
  if (!files.length) return;
  stopRealtime();
  selBatchFiles = files; selGradeFile = null; gradeMode = "batch";
  $("gradeFileInfo").textContent = `เลือกหลายไฟล์ (ชุด): ${files.length} ไฟล์`;
  setNote("gradeStatus", "กดปุ่ม 'ตรวจคำตอบ' เพื่อตรวจทั้งชุด", "warn");
  $("gradeProcessBtn").disabled = false;
  e.target.value = "";
};
$("gradeProcessBtn").onclick = async () => {
  if (!requireReady("gradeStatus") || !requireKey("gradeStatus")) return;
  stopRealtime();
  if (!(await ensureModel())) return;
  await withBusy("gradeProcessBtn", async () => {
    if (gradeMode === "batch" && selBatchFiles) { showBatch(); await gradeBatch(selBatchFiles.map(f => ({ file: f, name: f.name, realName: "", nick: "" }))); }
    else if (selGradeFile) { showSingle(); await gradeSingle(await fileToBGR(selGradeFile)); }
  });
};
$("gradeCamBtn").onclick = async () => {
  if (!requireReady("gradeStatus") || !requireKey("gradeStatus")) return;
  stopRealtime(); showSingle();
  if (!(await ensureModel())) return;
  const bgr = await openCamera();
  if (bgr) await withBusy(null, () => gradeSingle(bgr));
};

// ── โหมดเรียลไทม์ (กล้องสด) — ใช้ 1 คู่เสมอ ──
let rtActive = false, rtStream = null, rtLevel = 1;
function stopRealtime() {
  rtActive = false;
  if (rtStream) { rtStream.getTracks().forEach(t => t.stop()); rtStream = null; }
  if ($("rtCam")) $("rtCam").srcObject = null;
}
$("rtStopBtn").onclick = () => { stopRealtime(); showSingle(); setNote("gradeStatus", "ออกจากโหมดเรียลไทม์แล้ว", "ok"); setStatus("พร้อมใช้งาน"); };
$("rtFlipBtn").onclick = async () => {
  if (!rtStream) return;
  const next = camFacing === "environment" ? "user" : "environment";
  // ปิดกล้องเดิมก่อน (มือถือเปิดได้ทีละตัว) แล้วค่อยขอทิศใหม่
  rtStream.getTracks().forEach(t => t.stop()); rtStream = null; $("rtCam").srcObject = null;
  let s;
  try { s = await getCamStream(next); }
  catch (e) {
    setNote("gradeStatus", await camErrorMsg(e), "bad");
    try { rtStream = await getCamStream(camFacing); $("rtCam").srcObject = rtStream; } catch (_) {}
    return;
  }
  camFacing = next; rtStream = s; $("rtCam").srcObject = s; resetZoom($("rtCam"));
};

// ── แตะหน้าจอเพื่อโฟกัสกล้อง (รองรับเฉพาะเครื่อง/เบราว์เซอร์ที่มี focusMode + pointsOfInterest) ──
async function focusAt(stream, nx, ny) {
  if (!stream) return false;
  const track = stream.getVideoTracks()[0];
  if (!track || !track.getCapabilities) return false;
  const caps = track.getCapabilities();
  const adv = {};
  if (caps.pointsOfInterest) adv.pointsOfInterest = [{ x: nx, y: ny }];
  if (caps.focusMode) {
    if (caps.focusMode.includes("single-shot")) adv.focusMode = "single-shot";
    else if (caps.focusMode.includes("manual")) adv.focusMode = "manual";
    else if (caps.focusMode.includes("continuous")) adv.focusMode = "continuous";
  }
  if (!Object.keys(adv).length) return false;
  try { await track.applyConstraints({ advanced: [adv] }); return true; }
  catch { return false; }
}
function showFocusRing(x, y, ok) {
  const r = document.createElement("div");
  r.style.cssText = `position:fixed;left:${x - 32}px;top:${y - 32}px;width:64px;height:64px;` +
    `border:2px solid ${ok ? "#ffd400" : "#888"};border-radius:10px;z-index:9999;pointer-events:none;` +
    `box-shadow:0 0 0 1px rgba(0,0,0,.4);transition:transform .25s ease-out,opacity .5s;transform:scale(1.4);opacity:1`;
  document.body.appendChild(r);
  requestAnimationFrame(() => { r.style.transform = "scale(1)"; });
  setTimeout(() => { r.style.opacity = "0"; }, 350);
  setTimeout(() => r.remove(), 900);
}
function attachTapFocus(videoEl, getStream) {
  videoEl.style.cursor = "crosshair";
  videoEl.addEventListener("click", async e => {
    const rect = videoEl.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const nx = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const ny = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
    const ok = await focusAt(getStream(), nx, ny);
    showFocusRing(e.clientX, e.clientY, ok);
  });
}
attachTapFocus($("cam"), () => camStream);
attachTapFocus($("rtCam"), () => rtStream);
attachTapFocus($("capCam"), () => capStream);
attachPinchZoom($("cam"), () => camStream);
attachPinchZoom($("rtCam"), () => rtStream);
attachPinchZoom($("capCam"), () => capStream);

$("realtimeBtn").onclick = async () => {
  if (!requireReady("gradeStatus") || !requireKey("gradeStatus")) return;
  if (!(await ensureModel())) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setNote("gradeStatus", "เบราว์เซอร์นี้ใช้กล้องไม่ได้ — ต้องเปิดผ่าน https หรือ localhost", "bad"); return;
  }
  if (await camPermissionState() === "denied") {
    setNote("gradeStatus", await camErrorMsg({ name: "NotAllowedError" }), "bad"); return;
  }
  setNote("gradeStatus", "กำลังขอสิทธิ์ใช้กล้อง… โปรดกด 'อนุญาต'", "warn");
  try {
    rtStream = await getCamStream();
  } catch (e) { setNote("gradeStatus", await camErrorMsg(e), "bad"); return; }
  $("rtCam").srcObject = rtStream;
  resetZoom($("rtCam"));
  showRealtime();
  rtActive = true;
  setNote("gradeStatus", "เรียลไทม์: เล็งกระดาษให้เห็นครบ 4 คอลัมน์", "warn");
  rtLoop();
};

async function rtLoop() {
  while (rtActive) {
    const cam = $("rtCam");
    if (cam.videoWidth) {
      let bgr;
      try {
        bgr = imgElToBGR43(cam);   // crop 4:3
        const { results, colsMeta } = await grader.grade(bgr, rtLevel, null, P.RT_PAIRS);
        if (results.length && colsMeta.length === EXPECTED_COLS) {
          const { correct, graded } = P.scoreResults(results, key);
          const [match] = P.compareSignatures(keySig, P.classSignature(colsMeta));
          const cfq = P.colFirstQuestion(colsMeta);
          const canvas = P.annotate(colsMeta, key, cfq);
          showBGR("rtCanvas", canvas); canvas.delete();
          const pct = 100 * correct / Math.max(graded, 1);
          const sl = $("rtScore");
          sl.textContent = `${correct}/${graded}  (${pct.toFixed(1)}%)` + (match ? "" : " · กระดาษต่างเฉลย");
          sl.className = "score " + (!match ? "bad" : pct >= 50 ? "ok" : "bad");
        } else {
          const sl = $("rtScore"); sl.textContent = "เล็งกระดาษให้เห็นครบ 4 คอลัมน์…"; sl.className = "score warn";
        }
        if (colsMeta) colsMeta.forEach(m => m.img.delete());
      } catch (e) { console.error(e); }
      finally { if (bgr) bgr.delete(); }
    }
    await new Promise(r => setTimeout(r, 120));
  }
}

async function gradeSingle(bgr) {
  setNote("gradeStatus", "กำลังตรวจคำตอบ…", "warn"); setStatus("กำลังตรวจคำตอบ…", "warn");
  try {
    const { results, colsMeta } = await grader.grade(bgr, level, abortSignal());
    if (!results.length) { setNote("gradeStatus", "ตรวจไม่พบคอลัมน์ — " + DETECT_HINT, "bad"); bgr.delete(); return; }
    if (colsMeta.length !== EXPECTED_COLS) {
      const canvas = P.annotate(colsMeta);
      if (canvas) { showBGR("gradeCanvas", canvas); canvas.delete(); }
      const sl = $("scoreLbl"); sl.textContent = "—"; sl.className = "score bad";
      $("gradeFlags").innerHTML = "";
      setNote("gradeStatus", `detect ผิดพลาด: พบ ${colsMeta.length}/${EXPECTED_COLS} คอลัมน์ — ${DETECT_HINT}`, "bad");
      $("chkLbl").textContent = ""; $("wrongText").textContent = "—";
      colsMeta.forEach(m => m.img.delete()); bgr.delete(); return;
    }
    const { correct, graded, wrong } = P.scoreResults(results, key);
    const [match, msg] = P.compareSignatures(keySig, P.classSignature(colsMeta));
    const cfq = P.colFirstQuestion(colsMeta);
    const canvas = P.annotate(colsMeta, key, cfq);
    showBGR("gradeCanvas", canvas); canvas.delete();
    const pct = 100 * correct / Math.max(graded, 1);
    const sl = $("scoreLbl"); sl.textContent = `${correct}/${graded}  (${pct.toFixed(1)}%)`;
    sl.className = "score " + (pct >= 50 ? "ok" : "bad");
    renderFlags("gradeFlags", results);
    const cl = $("chkLbl");
    cl.textContent = (match ? "✓ " : "⚠ ") + msg;
    cl.className = "chk " + (match ? "ok" : "bad");
    const detail = flagDetailText(results);
    $("wrongText").innerHTML = wrongHTML(wrong, detail);
    if (!match) setNote("gradeStatus", "⚠ " + msg.split("\n")[0], "bad");
    else setNote("gradeStatus", `ตรวจคำตอบเสร็จ: ถูก ${correct}/${graded}`, "ok");
    setStatus("ตรวจคำตอบเสร็จ");
    colsMeta.forEach(m => m.img.delete());
  } catch (e) {
    if (e.name === "AbortError") { setNote("gradeStatus", "ยกเลิกการตรวจแล้ว", "warn"); setStatus("ยกเลิกแล้ว", "warn"); }
    else { setNote("gradeStatus", "ตรวจล้มเหลว: " + e, "bad"); console.error(e); }
  }
  bgr.delete();
}

// items: { name, realName?, nick?, file? | dataURL? }  (รับได้ทั้งเลือกไฟล์ และถ่ายจากกล้อง)
async function gradeBatch(items) {
  const grid = $("batchGrid"); grid.innerHTML = ""; batchResults = [];
  let aborted = false;
  for (let i = 0; i < items.length; i++) {
    if (abortSignal() && abortSignal().aborted) { aborted = true; break; }
    setNote("gradeStatus", `กำลังตรวจคำตอบ ${i + 1}/${items.length}…`, "warn"); setStatus(`กำลังตรวจคำตอบ ${i + 1}/${items.length}…`, "warn");
    const it = items[i];
    const r = { name: it.name, realName: it.realName || "", nick: it.nick || "", ok: false };
    try {
      const bgr = it.file ? await fileToBGR(it.file) : await dataURLToBGR(it.dataURL);
      const { results, colsMeta } = await grader.grade(bgr, level, abortSignal());
      if (!results.length) { r.errMsg = "ไม่พบคอลัมน์"; }
      else if (colsMeta.length !== EXPECTED_COLS) {
        r.errMsg = `detect ผิดพลาด (${colsMeta.length}/${EXPECTED_COLS} คอลัมน์)`;
        colsMeta.forEach(m => m.img.delete());
      } else {
        const { correct, graded, wrong } = P.scoreResults(results, key);
        const [match, classMsg] = P.compareSignatures(keySig, P.classSignature(colsMeta));
        const cfq = P.colFirstQuestion(colsMeta);
        const canvas = P.annotate(colsMeta, key, cfq);
        r.imgURL = bgrToDataURL(canvas); canvas.delete();
        colsMeta.forEach(m => m.img.delete());
        Object.assign(r, {
          ok: true, correct, graded, wrong,
          pct: 100 * correct / Math.max(graded, 1),
          unc: results.filter(x => x.confident === false).length,
          match, classMsg, detail: flagDetailText(results),
        });
      }
      bgr.delete();
    } catch (err) {
      if (err.name === "AbortError") { aborted = true; break; }
      r.errMsg = "ผิดพลาด"; console.error(err);
    }
    const idx = batchResults.push(r) - 1;
    const card = r.ok ? scoreCard(r, idx) : errCard(r, idx);
    card.style.animationDelay = (i % 12) * 0.04 + "s";
    r.cardEl = card;
    grid.appendChild(card);
  }
  if (aborted) { setNote("gradeStatus", `ยกเลิกแล้ว — ตรวจไป ${batchResults.length}/${items.length} ใบ`, "warn"); setStatus("ยกเลิกแล้ว", "warn"); }
  else { setNote("gradeStatus", `เสร็จ ${items.length} ใบ`, "ok"); setStatus(`เสร็จ ${items.length} ใบ`); }
}

function cardNameHTML(r) {
  const nm = [r.realName, r.nick && `(${r.nick})`].filter(Boolean).join(" ");
  return (nm ? `<div class="nm">${esc(nm)}</div>` : "") + `<div class="fname">${esc(r.name)}</div>`;
}
function scoreCard(r, idx) {
  const d = document.createElement("div"); d.className = "card";
  const col = r.pct >= 50 ? "var(--ok)" : "var(--bad)";
  d.innerHTML =
    cardNameHTML(r) +
    `<div class="pct" style="color:${col}">${r.pct.toFixed(1)}%</div>` +
    `<div class="det">ถูก ${r.correct}/${r.graded}${r.match ? "" : " · ✗กระดาษต่างเฉลย"}${r.unc ? ` · ไม่ชัดเจน ${r.unc}` : ""}</div>` +
    `<div class="ring"><i style="--w:${Math.min(r.pct, 100)}%;background:${col}"></i></div>`;
  d.onclick = () => openResultModal(idx);
  return d;
}
function errCard(r, idx) {
  const d = document.createElement("div"); d.className = "card err";
  d.innerHTML = cardNameHTML(r) + `<div class="pct">${esc(r.errMsg)}</div><div class="det">${DETECT_HINT}</div>`;
  d.onclick = () => openResultModal(idx);
  return d;
}
function esc(s) { return String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

// ── popup ผลรายใบ ──
let rmIdx = -1;
function openResultModal(idx) {
  const r = batchResults[idx]; if (!r) return;
  rmIdx = idx;
  $("rmFile").textContent = "ไฟล์: " + r.name;
  $("rmReal").value = r.realName; $("rmNick").value = r.nick;
  const img = $("rmImg");
  if (r.ok) {
    img.style.display = "block"; img.src = r.imgURL;
    const sl = $("rmScore"); sl.textContent = `${r.correct}/${r.graded}  (${r.pct.toFixed(1)}%)`;
    sl.className = "score " + (r.pct >= 50 ? "ok" : "bad");
    const cl = $("rmChk"); cl.textContent = (r.match ? "✓ " : "⚠ ") + (r.classMsg || "").split("\n")[0];
    cl.className = "chk " + (r.match ? "ok" : "bad");
    $("rmFlags").innerHTML = "";
    const add = (cls, txt) => { const s = document.createElement("span"); s.className = "flag " + cls; s.textContent = txt; $("rmFlags").appendChild(s); };
    if (r.unc) add("unc", `ไม่ชัดเจน ${r.unc} ข้อ`);
    if (!r.unc) add("good", "ทุกข้อชัดเจน");
    $("rmWrong").innerHTML = wrongHTML(r.wrong, r.detail);
  } else {
    img.style.display = "none"; img.removeAttribute("src");
    const sl = $("rmScore"); sl.textContent = r.errMsg; sl.className = "score bad";
    $("rmChk").textContent = DETECT_HINT; $("rmChk").className = "chk bad";
    $("rmFlags").innerHTML = ""; $("rmWrong").textContent = "—";
  }
  $("resultModal").style.display = "flex";
}
function closeResultModal() { $("resultModal").style.display = "none"; rmIdx = -1; }
function syncName() {
  const r = batchResults[rmIdx]; if (!r) return;
  r.realName = $("rmReal").value.trim(); r.nick = $("rmNick").value.trim();
  if (r.cardEl) {
    const old = r.cardEl.querySelector(".nm"); if (old) old.remove();
    const nm = [r.realName, r.nick && `(${r.nick})`].filter(Boolean).join(" ");
    if (nm) { const d = document.createElement("div"); d.className = "nm"; d.textContent = nm; r.cardEl.insertBefore(d, r.cardEl.firstChild); }
  }
}
$("rmReal").oninput = syncName;
$("rmNick").oninput = syncName;
$("resultClose").onclick = closeResultModal;
$("resultModal").onclick = e => { if (e.target === $("resultModal")) closeResultModal(); };
document.addEventListener("keydown", e => { if (e.key === "Escape" && $("resultModal").style.display === "flex") closeResultModal(); });

$("exportBtn").onclick = () => {
  if (!batchResults.length) { setNote("gradeStatus", "ยังไม่ได้ตรวจชุดกระดาษ", "bad"); return; }
  const head = ["file", "realname", "nick", "total", "correct", "wrong", "percent", "classchk"];
  const rows = batchResults.map(r => r.ok
    ? [r.name, r.realName, r.nick, r.graded, r.correct, r.graded - r.correct, r.pct.toFixed(1), r.match ? "ตรงเฉลย" : "ต่างจากเฉลย"]
    : [r.name, r.realName, r.nick, 0, 0, 0, "", r.errMsg]);
  const csv = [head].concat(rows).map(row => row.map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "batch_results.csv"; a.click();
};

// ── แกลเลอรีรูปตัวอย่างให้ดาวน์โหลดไปทดสอบ ──
const SAMPLE_IMAGES = "100-25.jpg,100-31.jpg,100-35.jpg,100-40.jpg,100-44.jpg,100-48.jpg,100-61.jpg,100-68.jpg,100-72.jpg,100-74.jpg,120-16.jpg,120-20.jpg,120-22.jpg,120-25.jpg,120-38.jpg,120-44.jpg,120-60.jpg,120-68.jpg,120-71.jpg,120-76.jpg,60-15.jpg,60-28.jpg,60-33.jpg,60-44.jpg,60-48.jpg,60-54.jpg,60-69.jpg,60-75.jpg,60-88.jpg,60-89.jpg,80-39.jpg,80-44.jpg,80-51.jpg,80-55.jpg,80-57.jpg,80-58.jpg,80-66.jpg,80-68.jpg,80-73.jpg,80-77.jpg".split(",");
function sampleCell(fn) {
  const src = "samples/" + fn;
  const cell = document.createElement("div");
  cell.style.cssText = "border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--panel)";
  const a = document.createElement("a");
  a.href = src; a.target = "_blank"; a.rel = "noopener";
  a.style.cssText = "display:block;line-height:0";
  const img = document.createElement("img");
  img.src = src; img.alt = fn; img.loading = "lazy";
  img.style.cssText = "width:100%;aspect-ratio:3/4;object-fit:cover;display:block";
  a.appendChild(img); cell.appendChild(a);
  const bar = document.createElement("div");
  bar.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:6px;padding:5px 8px";
  const lbl = document.createElement("span");
  lbl.textContent = fn; lbl.style.cssText = "font-size:12px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
  const dl = document.createElement("a");
  dl.href = src; dl.download = fn; dl.title = "ดาวน์โหลด"; dl.textContent = "⬇";
  dl.style.cssText = "text-decoration:none;color:var(--fg);font-size:15px;flex:0 0 auto";
  bar.appendChild(lbl); bar.appendChild(dl); cell.appendChild(bar);
  return cell;
}
(function buildGallery() {
  const g = $("sampleGallery"); if (!g) return;
  SAMPLE_IMAGES.slice(0, 4).forEach(fn => g.appendChild(sampleCell(fn)));
  const btn = $("sampleMoreBtn"), box = $("sampleAll"), grid = $("sampleAllGrid");
  if (!btn) return;
  let built = false;
  btn.onclick = () => {
    const open = box.style.display === "none";
    if (open && !built) { SAMPLE_IMAGES.forEach(fn => grid.appendChild(sampleCell(fn))); built = true; }
    box.style.display = open ? "block" : "none";
    btn.textContent = open ? "ซ่อน ▴" : "ดูทั้งหมด 40 ใบ ▾";
  };
})();

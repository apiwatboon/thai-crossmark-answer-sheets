// jstest.mjs — โหลดฟังก์ชันบริสุทธิ์จาก pipeline.js (shim window/cv/ort) แล้วเทียบกับ Python
import fs from "fs";

const code = fs.readFileSync(new URL("./pipeline.js", import.meta.url), "utf8");
const sandbox = { window: {}, cv: {}, ort: {} };
const fn = new Function("window", "cv", "ort", code + "\nreturn window.WebPipeline;");
const P = fn(sandbox.window, sandbox.cv, sandbox.ort);

// ดึงฟังก์ชันภายในที่ไม่ได้ export ผ่าน eval แยก (polyfit1, decodeE2E, orderColumns)
const internal = new Function("window", "cv", "ort",
  code + "\nreturn { polyfit1, decodeE2E, orderColumns, dedupCols, getPhysicalOrder };")(
  sandbox.window, sandbox.cv, sandbox.ort);

const data = JSON.parse(fs.readFileSync(new URL("./jstest_data.json", import.meta.url), "utf8"));

// 1) polyfit
const got = internal.polyfit1(data.polyfit.xv, data.polyfit.yv);
const exp = data.polyfit.expect;
const pfErr = Math.max(Math.abs(got[0] - exp[0]), Math.abs(got[1] - exp[1]));
console.log(`polyfit1: js=[${got[0].toFixed(5)}, ${got[1].toFixed(5)}]  numpy=[${exp[0].toFixed(5)}, ${exp[1].toFixed(5)}]  maxErr=${pfErr.toExponential(2)}`);

// 2) decode + ordering
const out = Float32Array.from(data.out);
const dets = internal.decodeE2E(out, data.dims, data.geo);
const ordered = internal.orderColumns(dets);
console.log(`\ndecode: js_cols=${ordered.length}  py_cols=${data.py_dets.length}`);
const labels = ["tl", "tr", "bl", "br"];
let maxKpErr = 0;
ordered.forEach(([ci, o], i) => {
  const p = data.py_dets[i];
  let line = `col${ci}: js cls=${o.cls} conf=${o.conf.toFixed(3)} | py cls=${p.cls} conf=${p.conf.toFixed(3)}`;
  for (const k of labels) {
    const e = Math.max(Math.abs(o[k][0] - p[k][0]), Math.abs(o[k][1] - p[k][1]));
    maxKpErr = Math.max(maxKpErr, e);
  }
  console.log(line);
});
console.log(`\nmax keypoint error (js vs py decode): ${maxKpErr.toExponential(2)} px`);
console.log(pfErr < 1e-6 && maxKpErr < 1e-3 && ordered.length === data.py_dets.length
  ? "\nPASS — pure-JS port ตรงกับ Python" : "\nFAIL — มีส่วนต่าง ตรวจสอบ");

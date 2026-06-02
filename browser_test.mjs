// browser_test.mjs — รันหน้าเว็บจริงแบบ headless แล้วเทียบคำตอบฝั่ง JS กับ py_ref.json
// ครอบทั้งสองรุ่นตั้งต้น (yolo26n_pose + yolo26s_pose) ด้วย grid-crop (expand+crop)
import fs from "fs";
import puppeteer from "puppeteer";

const PAGE_URL = "http://127.0.0.1:8000/index.html";
const pyRef = JSON.parse(fs.readFileSync(new URL("./py_ref.json", import.meta.url), "utf8"));
const models = Object.keys(pyRef); // ["yolo26n_pose", "yolo26s_pose"]

const browser = await puppeteer.launch({ headless: "new", args: ["--no-sandbox", "--disable-setuid-sandbox"] });
const page = await browser.newPage();
page.on("pageerror", e => console.log("  [pageerror]", e.message));

await page.goto(PAGE_URL, { waitUntil: "load", timeout: 60000 });
await page.waitForFunction(
  () => typeof cv !== "undefined" && cv.Mat && typeof ort !== "undefined" && window.WebPipeline,
  { timeout: 120000, polling: 500 });

const loadModel = (url) => page.evaluate(async (url) => {
  window.__g = await window.WebPipeline.WebGrader.load(url);
}, url);

const gradeOne = (file) => page.evaluate(async (file) => {
  const img = await new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = file; });
  const c = document.createElement("canvas"); c.width = img.naturalWidth; c.height = img.naturalHeight;
  c.getContext("2d").drawImage(img, 0, 0);
  const rgba = cv.imread(c); const bgr = new cv.Mat(); cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR); rgba.delete();
  const t0 = performance.now();
  const { results, colsMeta } = await window.__g.grade(bgr);
  const ms = Math.round(performance.now() - t0);
  colsMeta.forEach(m => m.img.delete()); bgr.delete();
  return { ms, answers: results.map(r => ({ q: r.question, a: r.answer })) };
}, file);

let grand = 0, grandN = 0;
for (const model of models) {
  console.log(`\n=== ${model} (grid-crop, reader=pixel_count) ===`);
  await loadModel("models/" + model + ".onnx");
  const sheets = Object.keys(pyRef[model]);
  let mSame = 0, mN = 0;
  for (const name of sheets) {
    const r = await gradeOne("t_" + name);
    const pyMap = new Map(pyRef[model][name].map(x => [x.question, x.answer]));
    let same = 0; const diff = [];
    for (const x of r.answers) { if (String(pyMap.get(x.q)) === String(x.a)) same++; else diff.push(`q${x.q}:js=${x.a}/py=${pyMap.get(x.q)}`); }
    mSame += same; mN += r.answers.length;
    console.log(`  ${name}: ${same}/${r.answers.length} ตรง  (${r.ms}ms)` + (diff.length ? "  ต่าง: " + diff.slice(0, 8).join(" ") : ""));
  }
  grand += mSame; grandN += mN;
  console.log(`  รวม ${model}: ${mSame}/${mN}` + (mSame === mN ? " — PASS" : " — มีส่วนต่าง"));
}
console.log(`\nรวมทั้งหมด: ${grand}/${grandN} ` + (grand === grandN ? "— PASS ทุกข้อตรง Python" : "— มีส่วนต่าง"));
await browser.close();

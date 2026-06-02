// tier_test.mjs — smoke-test ระดับโหวต 12/6/3/1 บนหน้าเว็บจริง (headless)
// โหลด EnsembleGrader ครั้งเดียว แล้ว grade แต่ละ tier เทียบจำนวนข้อ + ความมั่นใจ
import puppeteer from "puppeteer";

const PAGE_URL = "http://127.0.0.1:8000/index.html";
const SHEETS = ["t_100-26.jpg", "t_100-32.jpg"];
const LEVELS = [12, 6, 3, 1];

const browser = await puppeteer.launch({ headless: "new", args: ["--no-sandbox", "--disable-setuid-sandbox"] });
const page = await browser.newPage();
page.on("pageerror", e => console.log("  [pageerror]", e.message));
page.on("console", m => { if (m.type() === "error") console.log("  [console.error]", m.text()); });

await page.goto(PAGE_URL, { waitUntil: "load", timeout: 60000 });
await page.waitForFunction(
  () => typeof cv !== "undefined" && cv.Mat && typeof ort !== "undefined" && window.WebPipeline,
  { timeout: 120000, polling: 500 });

console.log("loading ensemble (6 sessions)…");
await page.waitForFunction(() => !!window.__eg || true, { timeout: 1000 }).catch(() => {});
await page.evaluate(async () => { window.__eg = await window.WebPipeline.EnsembleGrader.load(); });
console.log("ensemble loaded.");

const gradeOne = (file, level) => page.evaluate(async (file, level) => {
  const img = await new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = file; });
  const c = document.createElement("canvas"); c.width = img.naturalWidth; c.height = img.naturalHeight;
  c.getContext("2d").drawImage(img, 0, 0);
  const rgba = cv.imread(c); const bgr = new cv.Mat(); cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR); rgba.delete();
  const t0 = performance.now();
  const { results, colsMeta } = await window.__eg.grade(bgr, level);
  const ms = Math.round(performance.now() - t0);
  colsMeta.forEach(m => m.img.delete()); bgr.delete();
  const unc = results.filter(r => r.confident === false)
                     .map(r => `q${r.question}=${r.answer}(${r.votes}/${r.total})`);
  return { ms, n: results.length, total: results[0] ? results[0].total : 0, unc };
}, file, level);

for (const name of SHEETS) {
  console.log(`\n=== ${name} ===`);
  for (const level of LEVELS) {
    const r = await gradeOne(name, level);
    console.log(`  tier ${level}: ${r.n} ข้อ · ${r.total} วิธี/ข้อ · ไม่มั่นใจ ${r.unc.length} · ${r.ms}ms` +
      (r.unc.length ? "  " + r.unc.slice(0, 6).join(" ") : ""));
  }
}
await browser.close();
console.log("\ntier_test done.");

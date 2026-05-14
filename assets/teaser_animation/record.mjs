// Capture a deterministic PNG sequence of index.html (?record=1) using puppeteer-core
// against the system-installed Google Chrome.
import puppeteer from "puppeteer-core";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import { mkdirSync, rmSync, existsSync } from "node:fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const HTML   = pathToFileURL(join(__dirname, "index.html")).href + "?record=1";
const FRAMES = join(__dirname, "frames");
const FPS    = 30;
const TOTAL  = 46000; // ms — must match TOTAL_MS in index.html
const W = 1280, H = 720;
const FRAME_COUNT = Math.round((TOTAL / 1000) * FPS); // 1380

if (existsSync(FRAMES)) rmSync(FRAMES, { recursive: true, force: true });
mkdirSync(FRAMES, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  defaultViewport: { width: W, height: H, deviceScaleFactor: 1 },
  args: ["--font-render-hinting=none", "--disable-gpu", "--hide-scrollbars"],
});
const page = await browser.newPage();
await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
await page.goto(HTML, { waitUntil: "load" });
await page.waitForFunction("window.__ready === true", { timeout: 5000 });
// give fonts a beat to settle
await page.evaluate(() => document.fonts && document.fonts.ready);
await new Promise(r => setTimeout(r, 400));

const stageHandle = await page.$("#stage");

const t0 = Date.now();
for (let i = 0; i < FRAME_COUNT; i++) {
  const tMs = (i / FPS) * 1000;
  await page.evaluate(t => window.setT(t), tMs);
  const fname = join(FRAMES, String(i).padStart(4, "0") + ".png");
  await stageHandle.screenshot({ path: fname, omitBackground: false });
  if (i % 30 === 0) {
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    process.stdout.write(`\rframe ${i}/${FRAME_COUNT}  (elapsed ${elapsed}s)`);
  }
}
process.stdout.write(`\nrendered ${FRAME_COUNT} frames in ${((Date.now() - t0)/1000).toFixed(1)}s\n`);

await browser.close();

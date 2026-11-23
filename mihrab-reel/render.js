// Renders scene.html frame by frame in headless Chromium and encodes the reel with ffmpeg.
//
//   node render.js                    -> mihrab.mp4 (1080x1920, 30 fps, with audio from audio.py)
//   node render.js --still 12.5 [out] -> one PNG frame at t = 12.5 s
//   node render.js --from 0 --to 4    -> render only part of the timeline (quick previews)
const http = require("http");
const fs = require("fs");
const path = require("path");
const { spawn, execFileSync } = require("child_process");
const { chromium } = require("playwright");

const ROOT = __dirname;
const FPS = 30;
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".woff2": "font/woff2", ".json": "application/json" };
function serve() {
  return new Promise(resolve => {
    const srv = http.createServer((req, res) => {
      const file = path.join(ROOT, decodeURIComponent(req.url.split("?")[0]));
      if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end(); }
      res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" });
      fs.createReadStream(file).pipe(res);
    }).listen(0, "127.0.0.1", () => resolve(srv));
  });
}

(async () => {
  const srv = await serve();
  const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  page.on("console", m => { if (m.type() === "error" || m.type() === "warning") console.error("[page]", m.text()); });
  page.on("pageerror", e => console.error("[page error]", e.message));
  await page.goto(`http://127.0.0.1:${srv.address().port}/scene.html`);
  await page.evaluate(() => window.ready);
  const wrap = await page.$("#wrap");

  const still = opt("--still", null);
  if (still !== null) {
    const out = opt(still, null) || args[args.indexOf("--still") + 2] || path.join(ROOT, `still_${still}.png`);
    await page.evaluate(t => window.renderAt(t), parseFloat(still));
    await wrap.screenshot({ path: out });
    console.log(out);
    await browser.close(); srv.close();
    return;
  }

  const duration = await page.evaluate(() => window.DURATION);
  const from = parseFloat(opt("--from", 0)), to = parseFloat(opt("--to", duration));
  const out = opt("--out", path.join(ROOT, "mihrab.mp4"));
  const wav = path.join(ROOT, "audio.wav");
  const withAudio = from === 0 && to === duration;
  if (withAudio) execFileSync("python3", [path.join(ROOT, "audio.py"), wav], { stdio: "inherit" });

  const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(FPS), "-i", "-",
    ...(withAudio ? ["-i", wav] : []),
    "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-r", String(FPS),
    ...(withAudio ? ["-c:a", "aac", "-b:a", "192k", "-shortest"] : []),
    "-movflags", "+faststart", out], { stdio: ["pipe", "inherit", "inherit"] });

  const n0 = Math.round(from * FPS), n1 = Math.round(to * FPS);
  const t0 = Date.now();
  for (let i = n0; i < n1; i++) {
    await page.evaluate(t => window.renderAt(t), i / FPS);
    const buf = await wrap.screenshot({ type: "png" });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once("drain", r));
    if (i % 15 === 0) {
      const done = i - n0 + 1, per = (Date.now() - t0) / done / 1000;
      console.error(`frame ${i}/${n1}  ${per.toFixed(2)} s/frame  eta ${Math.round(per * (n1 - i))} s`);
    }
  }
  ff.stdin.end();
  await new Promise(r => ff.on("close", r));
  if (withAudio) fs.unlinkSync(wav);
  console.log("wrote", out);
  await browser.close(); srv.close();
})();

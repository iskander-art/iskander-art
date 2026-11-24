// Renders captions.html frame by frame (transparent) into a video with alpha.
//   node captions.js out.mov [duration]          -> PNG-in-MOV with alpha, 30 fps
//   node captions.js --still t out.png           -> one transparent frame
const http = require("http");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const { chromium } = require("playwright");

const ROOT = __dirname;
const TYPES = { ".html": "text/html", ".woff2": "font/woff2" };

function serve() {
  return new Promise(resolve => {
    const srv = http.createServer((req, res) => {
      const f = path.join(ROOT, decodeURIComponent(req.url.split("?")[0]));
      if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end(); }
      res.writeHead(200, { "Content-Type": TYPES[path.extname(f)] || "application/octet-stream" });
      fs.createReadStream(f).pipe(res);
    }).listen(0, "127.0.0.1", () => resolve(srv));
  });
}

(async () => {
  const args = process.argv.slice(2);
  const srv = await serve();
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  await page.goto(`http://127.0.0.1:${srv.address().port}/captions.html`);
  await page.evaluate(() => window.ready);
  if (args[0] === "--still") {
    await page.evaluate(t => window.renderAt(t), parseFloat(args[1]));
    await page.screenshot({ path: args[2], omitBackground: true });
  } else {
    const out = args[0], dur = parseFloat(args[1] || "28.8"), fps = 30;
    const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(fps), "-i", "-",
      "-c:v", "png", "-pix_fmt", "rgba", out], { stdio: ["pipe", "inherit", "inherit"] });
    const n = Math.round(dur * fps);
    for (let i = 0; i < n; i++) {
      await page.evaluate(t => window.renderAt(t), i / fps);
      const buf = await page.screenshot({ omitBackground: true, type: "png" });
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once("drain", r));
    }
    ff.stdin.end();
    await new Promise(r => ff.on("close", r));
    const times = await page.evaluate(() => window.CAP_TIMES);
    fs.writeFileSync(out + ".pops.json", JSON.stringify(times));
    console.log("wrote", out);
  }
  await browser.close();
  srv.close();
})();

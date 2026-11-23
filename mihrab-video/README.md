# MIHRAB — a reel made only with code

`mihrab.mp4` (1080×1920, 30 fps, 20 s) is generated entirely by `mihrab.py`: no images, footage or audio files are used.

| Time | What happens |
|------|--------------|
| 0–2.6 s | A gold point, then a compass-and-straightedge construction: circle, diameters, octagon |
| 2.6–8 s | The camera pulls back while the 8-pointed star strapwork grows outward |
| 6.6–9 s | Zellige tiles (cream, green, cobalt, honey, terracotta) pop in as a ripple |
| 8.6–13 s | The mihrab assembles: horseshoe arch outline, colonnettes, alternating voussoirs, alfiz frame, star frieze, shell hood |
| 12–20 s | A light beam with dust, then the title, typed subtitle and handle |

**How it works**
- **Pattern:** a 4.8.8 tiling (octagons and squares) turned into stars with Hankin's "polygons in contact" method (contact angle 67.5°).
- **Mihrab:** a pointed horseshoe arch made of two offset circular arcs, with concentric voussoirs.
- **Rendering:** pycairo draws each frame as vectors, and raw frames are piped straight into ffmpeg (H.264).
- **Sound:** numpy synthesis. A D/A drone, plus Karplus-Strong "oud" plucks in maqam Hijaz on D, timed to the picture, with convolution reverb.

## Run it

```bash
pip install pycairo numpy
python mihrab.py              # renders mihrab.mp4 (~2 min on 4 cores)
python mihrab.py --still 13   # renders a single PNG frame at t = 13 s
```

Colors, timings and the handle are constants and functions near the top of `mihrab.py`: `GOLD`, `GREEN`, `camera()`, `draw_text()` and so on.

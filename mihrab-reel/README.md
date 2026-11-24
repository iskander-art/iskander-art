# Mihrab reel, made only with code

`mihrab.mp4` (1080×1920, 30 fps, 20 s) is a product-style reel of the gold mirror mihrab panel. No photos, footage or recorded sound are used. Everything is built from code:
- the room, the panel, the calligraphy, the rug, the plant and the Quran stand;
- the light, the camera moves and the soundtrack.

The style follows the inspiration reel: a laser-cut piece on a clean white wall, styled with a prayer rug and plants, in soft natural light, with no text on screen.

| Time | Shot |
|------|------|
| 0–3.6 s | Close glide across the sunlit lattice and its shadow on the wall |
| 3.6–7.2 s | From under the arch, looking up at the onion tip, pulling back |
| 7.2–10.6 s | Pan across the calligraphy: محمد, then فَوَلِّ وَجْهَكَ شَطْرَ الْمَسْجِدِ الْحَرَامِ, then الله |
| 10.6–15.8 s | Three-quarter view of the corner: window, sheer curtain, plant, Quran stand, rug |
| 15.8–20 s | Front view, slow push-in, fade out |

## How it works
- **`scene.html`** builds the scene in three.js.
  - **Panel:** the outline is measured from the photos (1.10 × 1.70 m, stepped shoulders, onion arch with a pointed tip). Its lattice is a 4.6.12 tiling turned into 12-fold rosettes with Hankin's method. The lattice is drawn on a canvas, blurred slightly so the cut-outs get laser-rounded corners, and used as a cut-out mask on a mirror-gold metal material.
  - **Reflections:** the gold reflects a cube-map capture of the room itself.
  - **Sunlight:** comes through the window and casts the lattice and window shadows on the wall.
  - **Calligraphy:** set in the Amiri font on black acrylic with gold lattice rings.
- **`render.js`** opens the page in headless Chromium, renders each frame and pipes it into ffmpeg.
- **`audio.py`** synthesizes the soundtrack: a drone, oud-like plucks in maqam Hijaz, a frame drum on each cut, and reverb.

## Run it

```bash
npm install            # three.js + the Amiri font
pip install numpy
node render.js                     # full reel -> mihrab.mp4 (software WebGL, takes a while)
node render.js --still 13 out.png  # a single frame at t = 13 s
```

It needs Node, Playwright's Chromium, Python 3 and ffmpeg. To change the cuts or camera moves, edit `SHOTS` in `scene.html`. Panel dimensions are `PANEL_*` and the arch is `openingContour()`.

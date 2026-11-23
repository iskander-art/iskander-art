# Mihrab footage: realistic reel from three photos, code only

`footage.py` turns three still photos of the mihrab into a 20 s vertical reel (1080×1920, 30 fps, with sound). The result looks like handheld camera footage. No AI models are used, only Python (numpy and OpenCV) and ffmpeg.

| Time | Photo | Move |
|------|-------|------|
| 0–3.6 s | corner with window | dolly to the right + push-in; curtain and plant leaves move; dust in the sunlight |
| 3.6–7.2 s | side view, balcony door | sideways dolly: the building outside slides behind the door frame |
| 7.2–10.8 s | front view | tilt down from the calligraphy to the arch, with a glint on the gold |
| 10.8–14.4 s | side view | close-up drifting down the lattice and the arch shoulder |
| 14.4–20 s | front view | pull back to reveal the whole corner, then fade out |

## How it works
- **Depth:** each photo gets a hand-built depth map, with walls as tilted planes and the floor rising towards the camera. The door leaves, plant, Quran stand and armchairs are separate polygons. A backward warp shifts every pixel by its depth while the virtual camera moves, which creates real parallax.
- **Moving details:** local sine warps, masked to the leaves and the curtain, make them move.
- **Gold glint:** the mirror gold is found by color, restricted to the mihrab and calligraphy, and gets a moving highlight band.
- **Camera feel:** handheld micro-motion, sharpening scaled by zoom, highlight bloom, a warm grade, vignette and film grain.
- **Sound:** `audio.py` synthesizes the soundtrack (drone, plucks in maqam Hijaz, frame drum on each cut) aligned to the cuts.

## Run it

```bash
pip install numpy opencv-python-headless
python footage.py --photos side.jpg front.png corner.png            # -> mihrab.mp4
python footage.py --photos side.jpg front.png corner.png --still 9  # one frame
python footage.py --photos ... --debug-depth                        # depth maps + gold mask
```

The photos are not stored in this repository. The polygons in `scene_side`, `scene_front` and `scene_corner` are in the 941×1672 pixel space of the original photos. Larger photos are scaled to that size; use full-resolution originals and scale the coordinates for sharper close-ups.

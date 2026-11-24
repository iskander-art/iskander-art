# Mihrab footage: realistic reel from three photos, code only

`footage.py` turns three still photos of the mihrab into footage that looks handheld. No AI models are used, only Python (numpy and OpenCV) and ffmpeg. It produces:
- **an edited reel:** 11 shots, ~33 s, 1080×1920, 30 fps, with sound;
- **every shot as its own silent clip** (`--clips`), for editing in CapCut, Premiere and similar apps.

## Shot library

| # | Shot | Photo | Move |
|---|------|-------|------|
| 1 | `corner_dolly` | corner with window | dolly right + push-in; curtain and leaves move; dust in the sun |
| 2 | `calligraphy_pan` | front | pan from محمد across the cartouche to الله |
| 3 | `lattice_glide` | side | close-up drifting down the lattice and the arch shoulder |
| 4 | `door_reveal` | side | starts on the view through the balcony door, pulls back to reveal the mihrab |
| 5 | `plant_focus` | front | focus pull from the plant and Quran stand to the gold jamb |
| 6 | `arch_tilt` | front | tilt down from the calligraphy to the arch |
| 7 | `allah_close` | corner | slow push-in on the الله medallion, seen at an angle |
| 8 | `sunlit_jamb` | corner | tilt up the sunlit left jamb to the arch shoulder |
| 9 | `rug_rise` | corner | rises from the rug to the mihrab while the focus follows |
| 10 | `door_slide` | side | sideways dolly; the building outside slides behind the door frame |
| 11 | `front_pullback` | front | pull back to reveal the whole corner |

The reel order is `REEL` in `footage.py`, and each shot's length is its `dur`. Reorder or drop names to re-edit.

## How it works
- **Depth:** each photo gets a hand-built depth map, with walls as tilted planes and the floor rising towards the camera. The door leaves, plant, Quran stand and armchairs are separate polygons. A backward warp shifts every pixel by its depth while the virtual camera moves, which creates real parallax.
- **Focus pulls:** blur each pixel by its distance from a moving focus depth.
- **Moving details:** masked sine warps make the leaves and curtain move. The mirror gold, found by color and restricted to the mihrab, gets a moving glint.
- **Camera feel:** handheld micro-motion, sharpening scaled by zoom, bloom, a warm grade, vignette and film grain.
- **Sound:** `audio.py` synthesizes the soundtrack (drone, plucks in maqam Hijaz, frame drum on every cut).

## Run it

```bash
pip install numpy opencv-python-headless
python footage.py --photos side.jpg front.png corner.png                        # reel -> mihrab.mp4
python footage.py --photos side.jpg front.png corner.png --clips clips/         # reel + every shot
python footage.py --photos side.jpg front.png corner.png --clips clips/ --only-clips
python footage.py --photos side.jpg front.png corner.png --still 13             # one frame
python footage.py --photos ... --debug-depth                                    # depth maps + gold mask
```

The photos are not stored in this repository. The polygons and crops are in the 941×1672 pixel space of the photos. Larger photos are scaled to that size; use full-resolution originals and scale the coordinates for sharper close-ups.

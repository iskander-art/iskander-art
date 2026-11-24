"""
The Nour Gallery: organic mihrab reel (1080x1920, 30 fps).

Animates still images with code: parallax from simple depth maps (wall, floor, door, the
person in the abaya cut out by darkness), handheld micro-motion, a glint on the gold,
twinkling fairy lights and a flickering lantern. Shots are cut to the voiceover timings.

  python reel.py --img DIR --out video.mp4          # silent picture track
  python reel.py --img DIR --still 12.0             # one frame

DIR must contain: install1.jpg install2.jpg install3.png family.jpg kids_story.jpg
tea_story.jpg lights.jpg side.jpg front.png corner.png
"""
import argparse
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mihrab-footage"))
import footage  # noqa: E402  (depth builders for the three real photos)

OUT_W, OUT_H, FPS = 1080, 1920, 30
DURATION = 28.8
FLASH_AT = 24.30


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, a, b):
    return clamp((t - a) / (b - a))


def ease(x):
    return x * x * (3 - 2 * x)


def lerp(a, b, k):
    return a + (b - a) * k


# ----------------------------------------------------------------------------
# scenes
# ----------------------------------------------------------------------------
class Scene:
    """img, depth (0 far .. 1 near), sway, gold, twinkle, flicker maps, all in image space."""

    def __init__(self, img, depth, gold_box=None, sway=None, twinkle_box=None, flicker_box=None):
        self.img = img
        h, w = img.shape[:2]
        self.h, self.w = h, w
        self.depth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), max(3, w / 160))
        self.sway = np.zeros((h, w), np.float32) if sway is None else cv2.GaussianBlur(sway, (0, 0), 4)
        self.curtain = np.zeros((h, w), np.float32)
        self.gold = gold_mask(img, gold_box)
        self.twinkle, self.phase = twinkle_maps(img, twinkle_box)
        self.flicker = box_mask(h, w, flicker_box, blur=w / 60) if flicker_box else None


def box_mask(h, w, box, blur=6):
    m = np.zeros((h, w), np.float32)
    x0, y0, x1, y1 = [int(v) for v in box]
    m[y0:y1, x0:x1] = 1
    return cv2.GaussianBlur(m, (0, 0), blur)


def gold_mask(img, box):
    if box is None:
        return np.zeros(img.shape[:2], np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 14) & (h <= 33) & (s >= 110) & (v >= 120)).astype(np.float32)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), 1.0) * box_mask(*img.shape[:2], box)


def twinkle_maps(img, box):
    """Small bright blobs (fairy lights) inside box, plus a random phase per blob."""
    h, w = img.shape[:2]
    if box is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = max(5, int(w / 90) | 1)
    tophat = gray - cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = np.clip((tophat - 25) / 40, 0, 1) * (gray > 190) * box_mask(h, w, box, blur=2)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)
    rng = np.random.default_rng(9)
    phase = cv2.resize(rng.random((h // 12 + 1, w // 12 + 1)).astype(np.float32) * 6.283, (w, h), interpolation=cv2.INTER_NEAREST)
    return m, phase


def dark_figure(img, box, thresh=62):
    """Mask of the person in a black abaya: dark pixels inside a bounding box."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 2] < thresh).astype(np.uint8) * (box_mask(*img.shape[:2], box, blur=0.1) > 0.5)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m.astype(np.float32)


def room_depth(h, w, floor_y, outside=None, leaves=(), figures=(), fig_d=0.40):
    Y = np.mgrid[0:h, 0:w][0].astype(np.float32)
    d = np.full((h, w), 0.20, np.float32)
    fl = Y > floor_y
    d[fl] = 0.20 + 0.36 * (Y[fl] - floor_y) / (h - floor_y)
    if outside:
        x0, y0, x1, y1 = outside
        d[y0:y1, x0:x1] = 0.02
    for (x0, y0, x1, y1, v) in leaves:
        d[y0:y1, x0:x1] = v
    for m in figures:
        d = np.where(m > 0.5, fig_d, d)
    return d


def load(path, crop=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"cannot read {path}")
    if crop:
        x0, y0, x1, y1 = crop
        img = img[y0:y1, x0:x1].copy()
    return img


def build_scenes(d):
    S = {}
    # installation sequence (same room, AI-styled images)
    for name, fig_box in (("install1", (480, 590, 920, 1520)), ("install2", (590, 590, 930, 1520))):
        img = load(os.path.join(d, f"{name}.jpg"))
        fig = dark_figure(img, fig_box)
        dep = room_depth(*img.shape[:2], floor_y=1395, outside=(195, 455, 428, 1390),
                         leaves=[(0, 330, 190, 1460, 0.36), (428, 330, 466, 1440, 0.30)], figures=[fig])
        S[name] = Scene(img, dep, gold_box=(480, 480, 940, 1300), flicker_box=(985, 985, 1055, 1100))
    img = load(os.path.join(d, "install3.png"))
    fig = dark_figure(img, (330, 450, 800, 2050))
    dep = room_depth(*img.shape[:2], floor_y=1845, outside=(165, 700, 468, 1880),
                     leaves=[(0, 450, 165, 1980, 0.36)], figures=[fig])
    S["install3"] = Scene(img, dep, gold_box=(560, 470, 1360, 1800), flicker_box=(1320, 1300, 1420, 1440))
    # family moment
    img = load(os.path.join(d, "family.jpg"))
    mom = dark_figure(img, (290, 990, 700, 1570))
    dep = room_depth(*img.shape[:2], floor_y=1140, outside=(118, 450, 358, 1140),
                     leaves=[(0, 280, 118, 1230, 0.36)], figures=[mom], fig_d=0.48)
    kid = np.zeros_like(dep)
    kid[900:1290, 600:880] = 1
    dep = np.where(kid > 0, np.maximum(dep, 0.33), dep)
    S["family"] = Scene(img, dep, gold_box=(410, 190, 790, 960), flicker_box=(808, 812, 866, 915))
    # customer stories (the photo area only, without the burned-in text)
    img = load(os.path.join(d, "kids_story.jpg"), crop=(70, 450, 1010, 1750))
    h, w = img.shape[:2]
    dep = room_depth(h, w, floor_y=980)
    kids = np.zeros((h, w), np.float32)
    cv2.fillPoly(kids, [np.array([(60, 420), (330, 420), (360, 1300), (40, 1300)], np.int32)], 1)
    dep = np.where(kids > 0, 0.45, dep)
    S["kids"] = Scene(img, dep, gold_box=(420, 120, 830, 860), twinkle_box=(430, 140, 860, 960), flicker_box=(250, 760, 340, 880))
    img = load(os.path.join(d, "tea_story.jpg"), crop=(55, 415, 1025, 1630))
    h, w = img.shape[:2]
    dep = room_depth(h, w, floor_y=850)
    woman = np.zeros((h, w), np.float32)
    cv2.fillPoly(woman, [np.array([(0, 420), (200, 420), (330, 700), (640, 980), (640, 1215), (0, 1215)], np.int32)], 1)
    dep = np.where(woman > 0, 0.50, dep)
    S["tea"] = Scene(img, dep, gold_box=(420, 120, 840, 760), twinkle_box=(420, 150, 850, 820), flicker_box=(290, 600, 360, 700))
    img = load(os.path.join(d, "lights.jpg"))
    h, w = img.shape[:2]
    dep = room_depth(h, w, floor_y=1180)
    dep[700:1240, 30:420] = 0.30                                    # plant, lamp and side table
    S["lights"] = Scene(img, dep, gold_box=(400, 60, 960, 1080), twinkle_box=(420, 280, 960, 1120), flicker_box=(210, 780, 320, 900))
    # the three real photos (depth maps from the footage script)
    real = footage.load_scenes([os.path.join(d, "side.jpg"), os.path.join(d, "front.png"), os.path.join(d, "corner.png")])
    for name, sc in real.items():
        s = Scene.__new__(Scene)
        s.img, s.depth, s.sway, s.curtain, s.gold = sc.img, sc.depth, sc.sway, sc.curtain, sc.gold
        s.h, s.w = sc.img.shape[:2]
        s.twinkle = s.phase = s.flicker = None
        S[name] = s
    return S


# ----------------------------------------------------------------------------
# edit (times follow the voiceover)
# h = crop height in image px, c = crop centre, T = parallax shift (px at disparity 1)
# ----------------------------------------------------------------------------
SHOTS = [
    dict(t0=0.00, t1=1.95, sc="install1", h=(1960, 1760), c=((571, 1024), (610, 990)), T=((-22, 0), (22, 0)), glint=0.25),
    dict(t0=1.95, t1=4.35, sc="install2", h=(1900, 1700), c=((590, 1000), (640, 960)), T=((22, 0), (-22, 0)), glint=0.30),
    dict(t0=4.35, t1=5.70, sc="install3", h=(2500, 2200), c=((768, 1376), (800, 1300)), T=((-24, 0), (24, 0)), glint=0.40),
    dict(t0=5.70, t1=8.76, sc="side", h=(760, 680), c=((725, 640), (742, 905)), T=((-10, 0), (10, 0)), glint=0.65),
    dict(t0=8.76, t1=11.04, sc="corner", h=(470, 420), c=((752, 236), (754, 212)), T=((-6, 0), (6, 0)), glint=0.60),
    dict(t0=11.04, t1=14.42, sc="front", h=(520, 470), c=((330, 272), (305, 262)), T=((6, 0), (-6, 0)), glint=0.55),
    dict(t0=14.42, t1=18.34, sc="family", h=(1660, 1440), c=((470, 836), (500, 830)), T=((0, -16), (0, 16)), glint=0.30),
    dict(t0=18.34, t1=20.71, sc="kids", h=(1290, 1170), c=((395, 650), (425, 640)), T=((-14, 0), (14, 0)), glint=0.25),
    dict(t0=20.71, t1=22.27, sc="tea", h=(1210, 1080), c=((360, 605), (520, 580)), T=((14, 0), (-14, 0)), glint=0.25),
    dict(t0=22.27, t1=24.30, sc="corner", h=(1590, 1380), c=((470, 836), (505, 812)), T=((-28, 0), (28, 0)), glint=0.30),
    dict(t0=24.30, t1=DURATION, sc="lights", h=(1400, 1180), c=((600, 700), (650, 610)), T=((-14, 0), (14, 0)), glint=0.35),
]
CUTS = [s["t0"] for s in SHOTS[1:]]

U, V = np.meshgrid(np.arange(OUT_W, dtype=np.float32), np.arange(OUT_H, dtype=np.float32))
_rng = np.random.default_rng(4)
GRAIN = [_rng.normal(0, 1, (OUT_H // 2, OUT_W // 2)).astype(np.float32) for _ in range(8)]
VIGNETTE = (1 - 0.26 * np.clip(((U - OUT_W / 2) / (OUT_W * 0.62)) ** 2 + ((V - OUT_H / 2) / (OUT_H * 0.62)) ** 2, 0, 1) ** 1.3)[..., None]


def noise(t, seed):
    return math.sin(t * 1.7 + seed) * 0.6 + math.sin(t * 3.1 + seed * 2.3) * 0.3 + math.sin(t * 5.3 + seed * 0.7) * 0.1


def render(t, S):
    shot = next((s for s in SHOTS if s["t0"] <= t < s["t1"]), SHOTS[-1])
    sc = S[shot["sc"]]
    p = prog(t, shot["t0"], shot["t1"])
    k = ease(p) * 0.8 + p * 0.2
    crop_h = lerp(*shot["h"], k)
    cx = lerp(shot["c"][0][0], shot["c"][1][0], k) + noise(t, 1) * 1.6
    cy = lerp(shot["c"][0][1], shot["c"][1][1], k) + noise(t, 2) * 1.6
    tx = lerp(shot["T"][0][0], shot["T"][1][0], k)
    ty = lerp(shot["T"][0][1], shot["T"][1][1], k)
    rot = noise(t, 3) * 0.0016
    s = crop_h / OUT_H
    du, dv = (U - OUT_W / 2) * s, (V - OUT_H / 2) * s
    cr, sr = math.cos(rot), math.sin(rot)
    bx = (cx + du * cr - dv * sr).astype(np.float32)
    by = (cy + du * sr + dv * cr).astype(np.float32)
    mx, my = bx, by
    for _ in range(2):
        dd = cv2.remap(sc.depth, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mx, my = bx - tx * dd, by - ty * dd
    sw = cv2.remap(sc.sway, bx, by, cv2.INTER_LINEAR, borderValue=0)
    cu = cv2.remap(sc.curtain, bx, by, cv2.INTER_LINEAR, borderValue=0)
    mx = mx + sw * (2.2 * np.sin(t * 2.1 + by * 0.045) + np.sin(t * 3.7 + bx * 0.06)) + cu * 3.0 * np.sin(t * 1.3 + by * 0.012 + bx * 0.03)
    my = my + sw * 1.2 * np.sin(t * 2.6 + bx * 0.05)
    frame = cv2.remap(sc.img, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT).astype(np.float32)

    # gold glint
    gold = cv2.remap(sc.gold, mx, my, cv2.INTER_LINEAR, borderValue=0)[..., None]
    pos = lerp(-0.25, 1.25, p) * (OUT_W + OUT_H)
    band = np.exp(-(((U * 0.8 + V * 0.6) - pos) / 140.0) ** 2)[..., None]
    frame = frame + gold * band * shot["glint"] * np.array([170, 235, 255], np.float32)
    frame = frame * (1 + gold * 0.06 * math.sin(t * 2.0))
    # fairy lights twinkle
    if sc.twinkle is not None:
        tw = cv2.remap(sc.twinkle, mx, my, cv2.INTER_LINEAR, borderValue=0)
        ph = cv2.remap(sc.phase, mx, my, cv2.INTER_NEAREST, borderValue=0)
        amp = tw * (0.5 + 0.5 * np.sin(t * 4.2 + ph))
        glow = cv2.GaussianBlur(amp, (0, 0), 6)
        frame = frame + (amp * 70 + glow * 120)[..., None] * np.array([150, 220, 255], np.float32) / 255
        frame = frame * (1 - 0.18 * (tw * (0.5 - 0.5 * np.sin(t * 4.2 + ph))))[..., None]
    # lantern / lamp flicker
    if sc.flicker is not None:
        fl = cv2.remap(sc.flicker, mx, my, cv2.INTER_LINEAR, borderValue=0)
        f = 0.10 * math.sin(t * 11.0) + 0.06 * math.sin(t * 17.3 + 1) + 0.05 * math.sin(t * 5.1 + 2)
        frame = frame * (1 + (fl * f)[..., None])

    # sharpen, bloom, grade
    zoom = sc.h / crop_h
    blur = cv2.GaussianBlur(frame, (0, 0), 1.6)
    frame = frame + (frame - blur) * (0.30 + 0.35 * clamp((zoom - 1) / 1.2))
    bright = np.clip(frame - 205, 0, None)
    frame = frame + cv2.GaussianBlur(bright, (0, 0), 22) * 0.35
    frame = frame * np.array([0.97, 1.0, 1.03], np.float32)
    frame = 255 * (np.clip(frame, 0, None) / 255) ** 1.02
    frame = frame * VIGNETTE
    g = cv2.resize(GRAIN[int(t * FPS) % len(GRAIN)], (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    frame = frame + g[..., None] * 3.0
    # white flash on the brand cut, short fade from black at the start
    fl = max(0.0, 1 - abs(t - FLASH_AT) / 0.125) if t >= FLASH_AT - 0.125 else 0.0
    frame = frame * (1 - fl) + 255 * fl
    frame = frame * min(1.0, 0.25 + t / 0.25)
    return np.clip(frame, 0, 255).astype(np.uint8)


_S = None


def _init(d):
    global _S
    cv2.setNumThreads(1)
    _S = build_scenes(d)


def _frame(i):
    return render(i / FPS, _S).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", default="picture.mp4")
    ap.add_argument("--still", type=float)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if a.still is not None:
        S = build_scenes(a.img)
        p = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"still_{a.still:05.2f}.png")
        cv2.imwrite(p, render(a.still, S))
        print(p)
        return
    n = int(round(DURATION * FPS))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p", a.out],
                          stdin=subprocess.PIPE)
    with Pool(a.workers, initializer=_init, initargs=(a.img,)) as pool:
        for i, data in enumerate(pool.imap(_frame, range(n), chunksize=4)):
            ff.stdin.write(data)
            if i % 90 == 0:
                print(f"frame {i}/{n}", file=sys.stderr)
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()

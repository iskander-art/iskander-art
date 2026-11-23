"""
Turn three still photos of the mihrab into a realistic vertical reel, with code only.

For every photo a hand-built depth map (walls, floor, door, plant, furniture) lets a virtual
camera dolly, pan and push in with real parallax. Small local warps make the plant leaves and
the curtain move, a moving glint runs across the gold, and the shots are cut to a synthesized
soundtrack.

Usage:
  python footage.py --photos side.jpg front.png corner.png             # -> mihrab.mp4
  python footage.py --photos side.jpg front.png corner.png --still 9   # one frame as PNG
  python footage.py ... --debug-depth                                  # save the depth maps

Photo order: (1) side view with the open balcony door, (2) straight-on front view,
(3) three-quarter view of the corner with the window. Coordinates below are in the
941x1672 pixel space of those photos (they are rescaled to it if larger).
"""
import argparse
import math
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_W, OUT_H, FPS = 1080, 1920, 30
DURATION = 20.0
SRC_W, SRC_H = 941, 1672


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, a, b):
    return clamp((t - a) / (b - a))


def ease(x):
    return x * x * (3 - 2 * x)


def lerp(a, b, k):
    return a + (b - a) * k


def poly_mask(pts, h=SRC_H, w=SRC_W):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    return m > 0


GRID_Y, GRID_X = np.mgrid[0:SRC_H, 0:SRC_W].astype(np.float32)


def floor_below(line_pts):
    """Boolean mask of pixels below a polyline [(x, y), ...] spanning the image width."""
    xs = np.array([p[0] for p in line_pts], np.float32)
    ys = np.array([p[1] for p in line_pts], np.float32)
    y_line = np.interp(np.arange(SRC_W), xs, ys).astype(np.float32)
    return GRID_Y > y_line[None, :]


# ----------------------------------------------------------------------------
# depth (disparity: 0 = far, 1 = near) and motion masks for each photo
# ----------------------------------------------------------------------------
def scene_side():
    """Photo 1: open balcony door on the left, mihrab wall receding to the right."""
    X, Y = GRID_X, GRID_Y
    d = np.where(X >= 340, 0.20 + 0.20 * np.clip((X - 340) / 600, 0, 1), 0.22).astype(np.float32)
    d[(Y < 250) & (X < 340)] = 0.26                                     # ceiling
    d[poly_mask([(168, 488), (300, 488), (300, 1252), (195, 1252), (170, 1130)])] = 0.02   # view outside
    d[poly_mask([(262, 282), (332, 325), (358, 1345), (300, 1352), (300, 500), (262, 482)])] = 0.31  # right leaf
    d[poly_mask([(100, 455), (168, 470), (196, 1236), (150, 1252)])] = 0.36  # shutter leaf
    d[poly_mask([(0, 418), (100, 455), (150, 1252), (55, 1292), (0, 1262)])] = 0.48  # outer glass leaf
    floor = floor_below([(0, 1255), (339, 1268), (341, 1292), (941, 1502)])
    d[floor] = np.clip((Y[floor] - 1050) * 0.0009, 0.18, 1)
    sway = np.zeros_like(d)
    sway[poly_mask([(204, 645), (240, 650), (240, 895), (204, 895)])] = 0.6   # palm outside
    return d, sway, np.zeros_like(d)


def scene_front():
    """Photo 2: straight-on view of the wall."""
    X, Y = GRID_X, GRID_Y
    d = np.full((SRC_H, SRC_W), 0.20, np.float32)
    d[X < 135] = 0.23                                                   # curtain side
    floor = floor_below([(0, 1158), (941, 1152)])
    d[floor] = 0.20 + (Y[floor] - 1155) / 517 * 0.38
    plant = poly_mask([(0, 795), (85, 800), (178, 900), (178, 1000), (132, 1062), (125, 1105), (55, 1115), (0, 1105)])
    d[plant] = 0.26
    d[poly_mask([(92, 1035), (302, 1035), (302, 1218), (92, 1218)])] = 0.25   # rehal
    d[poly_mask([(0, 878), (48, 888), (48, 1342), (0, 1352)])] = 0.40          # armchair
    sway = (plant & (Y < 1060)).astype(np.float32)
    curtain = poly_mask([(0, 0), (135, 0), (140, 1150), (0, 1150)]).astype(np.float32) * np.clip(Y / 1150, 0, 1) ** 1.5
    return d, sway, curtain


def scene_corner():
    """Photo 3: three-quarter view, window and curtain on the left wall."""
    X, Y = GRID_X, GRID_Y
    d = (0.18 + 0.12 * np.clip((X - 245) / 625, 0, 1)).astype(np.float32)   # main wall, nearer on the right
    left = X < 245
    d[left] = 0.18 + 0.18 * (245 - X[left]) / 245                         # window wall comes towards us
    right = X > 872
    d[right] = 0.30 + 0.30 * (X[right] - 872) / 69                        # right wall
    floor = floor_below([(0, 1100), (245, 1066), (835, 1272), (941, 1214)])
    d[floor] = np.clip((Y[floor] - 700) * 0.00052, 0.18, 1)
    plant = poly_mask([(0, 848), (60, 858), (132, 878), (202, 930), (206, 1010), (152, 1062), (142, 1150),
                       (112, 1182), (80, 1252), (40, 1242), (0, 1202)])
    d[plant] = 0.27
    d[poly_mask([(184, 996), (332, 993), (332, 1142), (184, 1147)])] = 0.25       # rehal
    d[poly_mask([(0, 1095), (56, 1110), (56, 1342), (0, 1372)])] = 0.38          # armchair left
    d[poly_mask([(828, 1520), (941, 1440), (941, 1672), (818, 1672)])] = 0.58    # armchair right
    sway = (plant.astype(np.float32) * np.clip((1110 - Y) / 250, 0, 1))
    curtain = poly_mask([(0, 20), (232, 95), (246, 1110), (0, 1110)]).astype(np.float32) * np.clip(Y / 1110, 0, 1) ** 1.5
    return d, sway, curtain


def gold_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 14) & (h <= 33) & (s >= 110) & (v >= 120)).astype(np.float32)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), 1.0)


# ----------------------------------------------------------------------------
# the edit: (start, end, photo, crop height a->b, centre a->b, camera shift a->b, glint strength)
# crop heights / centres are in source pixels; shift is px of parallax at disparity 1.
# ----------------------------------------------------------------------------
SHOTS = [
    dict(t0=0.0, t1=3.6, photo="corner", h=(1590, 1380), c=((470, 836), (505, 812)), T=((-28, 0), (28, 0)), glint=0.22),
    dict(t0=3.6, t1=7.2, photo="side", h=(1540, 1440), c=((470, 836), (462, 830)), T=((26, 0), (-26, 0)), glint=0.25),
    dict(t0=7.2, t1=10.8, photo="front", h=(860, 790), c=((535, 440), (535, 720)), T=((0, -6), (0, 6)), glint=0.40),
    dict(t0=10.8, t1=14.4, photo="side", h=(760, 690), c=((725, 640), (742, 905)), T=((-10, 0), (10, 0)), glint=0.45),
    dict(t0=14.4, t1=20.0, photo="front", h=(1150, 1600), c=((505, 760), (470, 836)), T=((0, -16), (0, 16)), glint=0.25),
]
CUTS = [s["t0"] for s in SHOTS[1:]]


class Scene:
    def __init__(self, path, builder, gold_box):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"cannot read {path}")
        if img.shape[:2] != (SRC_H, SRC_W):
            img = cv2.resize(img, (SRC_W, SRC_H), interpolation=cv2.INTER_AREA)
        self.img = img
        d, sway, curtain = builder()
        self.depth = cv2.GaussianBlur(d, (0, 0), 6)
        self.sway = cv2.GaussianBlur(sway, (0, 0), 4)
        self.curtain = cv2.GaussianBlur(curtain, (0, 0), 8)
        box = np.zeros((SRC_H, SRC_W), np.float32)
        x0, y0, x1, y1 = gold_box                      # glint only on the mihrab + calligraphy
        box[y0:y1, x0:x1] = 1
        self.gold = gold_mask(img) * cv2.GaussianBlur(box, (0, 0), 6)


def smooth_noise(t, seed):
    return (math.sin(t * 1.7 + seed) * 0.6 + math.sin(t * 3.1 + seed * 2.3) * 0.3 + math.sin(t * 5.3 + seed * 0.7) * 0.1)


U, V = np.meshgrid(np.arange(OUT_W, dtype=np.float32), np.arange(OUT_H, dtype=np.float32))
rng = np.random.default_rng(4)
GRAIN = [rng.normal(0, 1, (OUT_H // 2, OUT_W // 2)).astype(np.float32) for _ in range(8)]
VIGNETTE = (1 - 0.28 * np.clip(((U - OUT_W / 2) / (OUT_W * 0.62)) ** 2 + ((V - OUT_H / 2) / (OUT_H * 0.62)) ** 2, 0, 1) ** 1.3)[..., None]
DUST = rng.random((120, 5))


def render(t, scenes):
    shot = next((s for s in SHOTS if s["t0"] <= t < s["t1"]), SHOTS[-1])
    sc = scenes[shot["photo"]]
    k = ease(prog(t, shot["t0"], shot["t1"])) * 0.8 + prog(t, shot["t0"], shot["t1"]) * 0.2
    crop_h = lerp(*shot["h"], k)
    cx = lerp(shot["c"][0][0], shot["c"][1][0], k)
    cy = lerp(shot["c"][0][1], shot["c"][1][1], k)
    tx = lerp(shot["T"][0][0], shot["T"][1][0], k)
    ty = lerp(shot["T"][0][1], shot["T"][1][1], k)
    # handheld micro-motion
    cx += smooth_noise(t, 1) * 1.6
    cy += smooth_noise(t, 2) * 1.6
    rot = smooth_noise(t, 3) * 0.0016
    s = crop_h / OUT_H
    du, dv = (U - OUT_W / 2) * s, (V - OUT_H / 2) * s
    cr, sr = math.cos(rot), math.sin(rot)
    bx = (cx + du * cr - dv * sr).astype(np.float32)
    by = (cy + du * sr + dv * cr).astype(np.float32)
    # parallax: shift by disparity (two fixed-point iterations of a backward warp)
    mx, my = bx, by
    for _ in range(2):
        dd = cv2.remap(sc.depth, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mx, my = bx - tx * dd, by - ty * dd
    # living details: leaves and curtain
    sw = cv2.remap(sc.sway, bx, by, cv2.INTER_LINEAR, borderValue=0)
    cu = cv2.remap(sc.curtain, bx, by, cv2.INTER_LINEAR, borderValue=0)
    mx = mx + sw * (2.2 * np.sin(t * 2.1 + by * 0.045) + 1.0 * np.sin(t * 3.7 + bx * 0.06)) \
            + cu * 3.0 * np.sin(t * 1.3 + by * 0.012 + bx * 0.03)
    my = my + sw * 1.2 * np.sin(t * 2.6 + bx * 0.05)
    frame = cv2.remap(sc.img, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    gold = cv2.remap(sc.gold, mx, my, cv2.INTER_LINEAR, borderValue=0)[..., None]

    # moving glint across the mirror gold
    local = prog(t, shot["t0"], shot["t1"])
    pos = lerp(-0.25, 1.25, local) * (OUT_W + OUT_H)
    band = np.exp(-(((U * 0.8 + V * 0.6) - pos) / 140.0) ** 2)[..., None]
    glow = gold * band * shot["glint"]
    frame = frame + glow * np.array([170, 235, 255], np.float32)          # BGR warm white
    frame = frame * (1 + gold * 0.06 * math.sin(t * 2.0))                 # gentle breathing of the reflection

    # sharpen (more when zoomed in), bloom, grade
    zoom = SRC_H / crop_h
    blur = cv2.GaussianBlur(frame, (0, 0), 1.6)
    frame = frame + (frame - blur) * (0.35 + 0.35 * clamp((zoom - 1) / 1.2))
    bright = np.clip(frame - 205, 0, None)
    frame = frame + cv2.GaussianBlur(bright, (0, 0), 22) * 0.35
    frame = frame * np.array([0.97, 1.0, 1.03], np.float32)               # a touch warmer
    frame = 255 * (np.clip(frame, 0, None) / 255) ** 1.02
    frame = frame * (1 + 0.012 * math.sin(t * 0.9))                       # light breathing

    # dust motes catching the window light (wide shots only)
    if shot["photo"] in ("corner", "side") and zoom < 1.4:
        lum = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        overlay = np.zeros((OUT_H, OUT_W), np.float32)
        for x, y, sp, sz, ph in DUST:
            px = (x * OUT_W + math.sin(t * (0.4 + sp) + ph * 6) * 30) % OUT_W
            py = (y * OUT_H - t * (10 + 25 * sp)) % OUT_H
            if lum[int(py), int(px)] < 175:
                continue
            r = 1.5 + 2.5 * sz
            cv2.circle(overlay, (int(px), int(py)), int(r), 1.0, -1, lineType=cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (0, 0), 1.6)
        frame = frame + overlay[..., None] * 70 * (0.5 + 0.5 * math.sin(t * 2))

    frame = frame * VIGNETTE
    g = cv2.resize(GRAIN[int(t * FPS) % len(GRAIN)], (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    frame = frame + g[..., None] * 3.2
    fade = min(prog(t, 0, 0.45), 1 - prog(t, DURATION - 0.9, DURATION))
    return np.clip(frame * fade, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", nargs=3, required=True, metavar=("SIDE", "FRONT", "CORNER"))
    ap.add_argument("--out", default=os.path.join(HERE, "mihrab.mp4"))
    ap.add_argument("--still", type=float)
    ap.add_argument("--debug-depth", action="store_true")
    args = ap.parse_args()
    scenes = {"side": Scene(args.photos[0], scene_side, (400, 60, 900, 1310)),
              "front": Scene(args.photos[1], scene_front, (220, 120, 850, 1075)),
              "corner": Scene(args.photos[2], scene_corner, (320, 85, 845, 1060))}
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if args.debug_depth:
        for name, sc in scenes.items():
            vis = (sc.depth * 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA)
            cv2.imwrite(os.path.join(out_dir, f"depth_{name}.png"), np.hstack([sc.img, vis, (np.dstack([sc.gold] * 3) * 255).astype(np.uint8)]))
        return
    if args.still is not None:
        p = os.path.join(out_dir, f"still_{args.still:05.2f}.png")
        cv2.imwrite(p, render(args.still, scenes))
        print(p)
        return
    wav = os.path.join(out_dir, "audio.wav")
    subprocess.run([sys.executable, os.path.join(HERE, "audio.py"), wav, ",".join(str(c) for c in CUTS), str(DURATION)], check=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-", "-i", wav,
                           "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", args.out],
                          stdin=subprocess.PIPE)
    n = int(DURATION * FPS)
    for i in range(n):
        ff.stdin.write(render(i / FPS, scenes).tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{n}", file=sys.stderr)
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

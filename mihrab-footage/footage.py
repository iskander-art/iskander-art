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
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_W, OUT_H, FPS = 1080, 1920, 30
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
# Shot library. Crop heights / centres are in source pixels (941x1672 photos);
# T is the camera shift in px of parallax at disparity 1; glint is the gold highlight;
# dof = (focus from, focus to, blur per unit of disparity) for a focus pull.
# ----------------------------------------------------------------------------
SHOTS = {
    "corner_dolly":    dict(photo="corner", dur=3.0, h=(1590, 1380), c=((470, 836), (505, 812)), T=((-28, 0), (28, 0)), glint=0.22),
    "calligraphy_pan": dict(photo="front", dur=3.0, h=(565, 545), c=((330, 285), (690, 280)), T=((8, 0), (-8, 0)), glint=0.55),
    "lattice_glide":   dict(photo="side", dur=2.6, h=(760, 690), c=((725, 640), (742, 905)), T=((-10, 0), (10, 0)), glint=0.45),
    "door_reveal":     dict(photo="side", dur=3.2, h=(640, 1300), c=((240, 760), (430, 836)), T=((12, 0), (-12, 0)), glint=0.20),
    "plant_focus":     dict(photo="front", dur=3.0, h=(700, 640), c=((200, 990), (215, 985)), T=((10, 0), (-10, 0)), glint=0.30,
                            dof=(0.26, 0.20, 75)),
    "arch_tilt":       dict(photo="front", dur=2.8, h=(860, 790), c=((535, 440), (535, 720)), T=((0, -6), (0, 6)), glint=0.40),
    "allah_close":     dict(photo="corner", dur=2.4, h=(470, 420), c=((752, 236), (754, 212)), T=((-6, 0), (6, 0)), glint=0.60),
    "sunlit_jamb":     dict(photo="corner", dur=2.6, h=(700, 660), c=((455, 800), (480, 560)), T=((-8, 0), (8, 0)), glint=0.45),
    "rug_rise":        dict(photo="corner", dur=3.2, h=(900, 1000), c=((420, 1210), (520, 700)), T=((0, -20), (0, 20)), glint=0.25,
                            dof=(0.36, 0.20, 20)),
    "door_slide":      dict(photo="side", dur=2.8, h=(1540, 1440), c=((470, 836), (462, 830)), T=((26, 0), (-26, 0)), glint=0.25),
    "front_pullback":  dict(photo="front", dur=4.2, h=(1150, 1600), c=((505, 760), (470, 836)), T=((0, -16), (0, 16)), glint=0.25),
}
REEL = ["corner_dolly", "calligraphy_pan", "lattice_glide", "door_reveal", "plant_focus", "arch_tilt",
        "allah_close", "sunlit_jamb", "rug_rise", "door_slide", "front_pullback"]


def timeline():
    t, out = 0.0, []
    for name in REEL:
        out.append((t, t + SHOTS[name]["dur"], name))
        t += SHOTS[name]["dur"]
    return out


TIMELINE = timeline()
DURATION = TIMELINE[-1][1]
CUTS = [round(t0, 3) for t0, _, _ in TIMELINE[1:]]


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


def load_scenes(photos):
    return {"side": Scene(photos[0], scene_side, (400, 60, 900, 1310)),
            "front": Scene(photos[1], scene_front, (220, 120, 850, 1075)),
            "corner": Scene(photos[2], scene_corner, (320, 85, 845, 1060))}


def smooth_noise(t, seed):
    return (math.sin(t * 1.7 + seed) * 0.6 + math.sin(t * 3.1 + seed * 2.3) * 0.3 + math.sin(t * 5.3 + seed * 0.7) * 0.1)


U, V = np.meshgrid(np.arange(OUT_W, dtype=np.float32), np.arange(OUT_H, dtype=np.float32))
rng = np.random.default_rng(4)
GRAIN = [rng.normal(0, 1, (OUT_H // 2, OUT_W // 2)).astype(np.float32) for _ in range(8)]
VIGNETTE = (1 - 0.28 * np.clip(((U - OUT_W / 2) / (OUT_W * 0.62)) ** 2 + ((V - OUT_H / 2) / (OUT_H * 0.62)) ** 2, 0, 1) ** 1.3)[..., None]
DUST = rng.random((120, 5))


def render_shot(shot, lt, t, sc):
    """Render shot `shot` at local time lt (s); t is a global clock for the living details."""
    p = clamp(lt / shot["dur"])
    k = ease(p) * 0.8 + p * 0.2
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
    pos = lerp(-0.25, 1.25, p) * (OUT_W + OUT_H)
    band = np.exp(-(((U * 0.8 + V * 0.6) - pos) / 140.0) ** 2)[..., None]
    frame = frame + gold * band * shot["glint"] * np.array([170, 235, 255], np.float32)   # BGR warm white
    frame = frame * (1 + gold * 0.06 * math.sin(t * 2.0))

    # sharpen (more when zoomed in)
    zoom = SRC_H / crop_h
    blur = cv2.GaussianBlur(frame, (0, 0), 1.6)
    frame = frame + (frame - blur) * (0.35 + 0.35 * clamp((zoom - 1) / 1.2))

    # depth of field / focus pull
    if shot.get("dof"):
        fa, fb, kk = shot["dof"]
        focus = lerp(fa, fb, ease(clamp((p - 0.25) / 0.5)))
        dfin = cv2.remap(sc.depth, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        coc = np.clip(np.abs(dfin - focus) * kk, 0, 7)[..., None]
        b1 = cv2.GaussianBlur(frame, (0, 0), 2.5)
        b2 = cv2.GaussianBlur(frame, (0, 0), 7.0)
        w1 = np.clip(coc / 2.5, 0, 1)
        w2 = np.clip((coc - 2.5) / 4.5, 0, 1)
        frame = frame * (1 - w1) + (b1 * (1 - w2) + b2 * w2) * w1

    # bloom + grade
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
            cv2.circle(overlay, (int(px), int(py)), int(1.5 + 2.5 * sz), 1.0, -1, lineType=cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (0, 0), 1.6)
        frame = frame + overlay[..., None] * 70 * (0.5 + 0.5 * math.sin(t * 2))

    frame = frame * VIGNETTE
    g = cv2.resize(GRAIN[int(t * FPS) % len(GRAIN)], (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    return frame + g[..., None] * 3.2


def render(t, scenes):
    """Frame of the full reel at time t."""
    t0, _, name = next((x for x in TIMELINE if x[0] <= t < x[1]), TIMELINE[-1])
    shot = SHOTS[name]
    frame = render_shot(shot, t - t0, t, scenes[shot["photo"]])
    fade = min(prog(t, 0, 0.45), 1 - prog(t, DURATION - 0.9, DURATION))
    return np.clip(frame * fade, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# parallel rendering
# ----------------------------------------------------------------------------
_SCENES = None


def _init(photos):
    global _SCENES
    cv2.setNumThreads(1)
    _SCENES = load_scenes(photos)


def _reel_frame(i):
    return render(i / FPS, _SCENES).tobytes()


def _clip_frame(job):
    name, i = job
    shot = SHOTS[name]
    return np.clip(render_shot(shot, i / FPS, i / FPS, _SCENES[shot["photo"]]), 0, 255).astype(np.uint8).tobytes()


def encode(path, frames, wav=None):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-"]
    if wav:
        cmd += ["-i", wav]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p"]
    cmd += (["-c:a", "aac", "-b:a", "192k", "-shortest"] if wav else ["-an"])
    cmd += ["-movflags", "+faststart", path]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for data in frames:
        ff.stdin.write(data)
    ff.stdin.close()
    ff.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", nargs=3, required=True, metavar=("SIDE", "FRONT", "CORNER"))
    ap.add_argument("--out", default=os.path.join(HERE, "mihrab.mp4"), help="the edited reel")
    ap.add_argument("--clips", help="also export every shot as its own silent clip into this folder")
    ap.add_argument("--only-clips", action="store_true", help="skip the reel")
    ap.add_argument("--still", type=float, help="render one frame of the reel at this time")
    ap.add_argument("--debug-depth", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.out))

    if args.debug_depth or args.still is not None:
        scenes = load_scenes(args.photos)
        if args.debug_depth:
            for name, sc in scenes.items():
                vis = cv2.applyColorMap((sc.depth * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
                cv2.imwrite(os.path.join(out_dir, f"depth_{name}.png"),
                            np.hstack([sc.img, vis, (np.dstack([sc.gold] * 3) * 255).astype(np.uint8)]))
        else:
            p = os.path.join(out_dir, f"still_{args.still:05.2f}.png")
            cv2.imwrite(p, render(args.still, scenes))
            print(p)
        return

    with Pool(args.workers, initializer=_init, initargs=(args.photos,)) as pool:
        if not args.only_clips:
            wav = os.path.join(out_dir, "audio.wav")
            subprocess.run([sys.executable, os.path.join(HERE, "audio.py"), wav,
                            ",".join(str(c) for c in CUTS), str(DURATION)], check=True)
            n = int(round(DURATION * FPS))
            print(f"reel: {len(REEL)} shots, {DURATION:.1f} s, {n} frames", file=sys.stderr)
            encode(args.out, pool.imap(_reel_frame, range(n), chunksize=4), wav)
            os.remove(wav)
            print("wrote", args.out)
        if args.clips:
            os.makedirs(args.clips, exist_ok=True)
            for idx, (name, shot) in enumerate(SHOTS.items(), 1):
                n = int(round(shot["dur"] * FPS))
                path = os.path.join(args.clips, f"{idx:02d}_{name}.mp4")
                encode(path, pool.imap(_clip_frame, [(name, i) for i in range(n)], chunksize=4))
                print("wrote", path)


if __name__ == "__main__":
    main()

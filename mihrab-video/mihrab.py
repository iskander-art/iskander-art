"""
MIHRAB — a reel generated entirely from code.

Every pixel and every sound in the video is computed here:
  * geometry  : 4.8.8 tiling + Hankin's "polygons in contact" method -> zellige pattern
  * mihrab    : pointed horseshoe arch, alternating voussoirs, alfiz frame, shell hood
  * rendering : pycairo (vector drawing) -> raw frames piped into ffmpeg
  * audio     : numpy synthesis (drone + Karplus-Strong plucks in maqam Hijaz)

Usage:
  python mihrab.py                 # render mihrab.mp4 (1080x1920, 30 fps)
  python mihrab.py --still 12.5    # render a single PNG at t=12.5 s (for previews)
"""

import argparse
import math
import os
import subprocess
import sys
import wave
from multiprocessing import Pool

import cairo
import numpy as np

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
W, H = 1080, 1920
FPS = 30
DURATION = 20.0
HERE = os.path.dirname(os.path.abspath(__file__))

# Palette (Maghrebi zellige + gold)
NIGHT = (0.035, 0.050, 0.090)
NIGHT2 = (0.060, 0.090, 0.150)
GOLD = (0.89, 0.71, 0.36)
GOLD_HI = (1.00, 0.93, 0.70)
CREAM = (0.94, 0.90, 0.80)
GREEN = (0.05, 0.36, 0.30)
COBALT = (0.10, 0.22, 0.52)
HONEY = (0.78, 0.52, 0.18)
TERRA = (0.62, 0.20, 0.14)
INK = (0.06, 0.06, 0.08)

# Pattern geometry (world units == pixels at zoom 1)
EDGE = 30.0                          # edge length of the octagons / squares
PITCH = EDGE * (1 + math.sqrt(2))    # distance between octagon centres
THETA = math.radians(67.5)           # Hankin contact angle

# Mihrab geometry (world coords, y up, origin = screen centre)
ARCH_R = 330.0          # radius of the two arch arcs
ARCH_E = 55.0           # horizontal offset of the arc centres -> pointed apex
ARCH_YC = 120.0         # height of the arc centres
ARCH_ALPHA = math.radians(24)  # how far the horseshoe returns below the centres
BASE_Y = -600.0         # floor of the niche
VOUSSOIR = 62.0         # thickness of the voussoir band
ALFIZ_PAD = 70.0        # frame padding around the voussoirs


# ----------------------------------------------------------------------------
# Easing / timing helpers
# ----------------------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def prog(t, t0, t1):
    return clamp((t - t0) / (t1 - t0))


def ease_io(x):
    return x * x * (3 - 2 * x)


def ease_out(x):
    return 1 - (1 - x) ** 3


def ease_io_cubic(x):
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def mix(a, b, k):
    return tuple(a[i] + (b[i] - a[i]) * k for i in range(len(a)))


# ----------------------------------------------------------------------------
# Geometry: 4.8.8 tiling + Hankin's method
# ----------------------------------------------------------------------------
def regular_polygon(cx, cy, n, circumradius, rot):
    return [(cx + circumradius * math.cos(rot + 2 * math.pi * k / n),
             cy + circumradius * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


def rotate(v, a):
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def intersect(p, d, q, e):
    """Intersection of lines p + s*d and q + u*e."""
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9:
        return None
    s = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return (p[0] + s * d[0], p[1] + s * d[1])


def hankin(poly, theta):
    """Return the star polygon [m0, X0, m1, X1, ...] for a CCW polygon."""
    n = len(poly)
    mids, dirs = [], []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        mids.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        dirs.append(((b[0] - a[0]) / L, (b[1] - a[1]) / L))
    star = []
    for i in range(n):
        j = (i + 1) % n
        d1 = rotate(dirs[i], theta)                       # forward + inward
        d2 = rotate((-dirs[j][0], -dirs[j][1]), -theta)   # backward + inward
        x = intersect(mids[i], d1, mids[j], d2)
        star += [mids[i], x]
    return star


class Tile:
    __slots__ = ("kind", "center", "poly", "star", "dist", "seed")

    def __init__(self, kind, center, poly, star, seed):
        self.kind, self.center, self.poly, self.star = kind, center, poly, star
        self.dist = math.hypot(*center)
        self.seed = seed


def build_pattern(extent_x, extent_y):
    tiles = []
    R_oct = EDGE / (2 * math.sin(math.pi / 8))
    R_sq = EDGE / math.sqrt(2)
    nx = int(extent_x / PITCH) + 2
    ny = int(extent_y / PITCH) + 2
    rng = np.random.default_rng(7)
    for i in range(-nx, nx + 1):
        for j in range(-ny, ny + 1):
            cx, cy = i * PITCH, j * PITCH
            oct_ = regular_polygon(cx, cy, 8, R_oct, math.pi / 8)
            tiles.append(Tile("oct", (cx, cy), oct_, hankin(oct_, THETA), rng.random()))
            sx, sy = cx + PITCH / 2, cy + PITCH / 2
            sq = regular_polygon(sx, sy, 4, R_sq, 0.0)
            tiles.append(Tile("sq", (sx, sy), sq, hankin(sq, THETA), rng.random()))
    tiles.sort(key=lambda t: t.dist)
    return tiles


# ----------------------------------------------------------------------------
# Geometry: the mihrab
# ----------------------------------------------------------------------------
def arc_pts(cx, cy, r, a0, a1, n):
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / (n - 1)),
             cy + r * math.sin(a0 + (a1 - a0) * k / (n - 1))) for k in range(n)]


def arch_halves(offset=0.0, n=90):
    """Left and right arch curves (springer -> apex), concentric offset outward."""
    r = ARCH_R + offset
    phi = math.acos(ARCH_E / r)
    # left half is centred on the RIGHT centre, and vice versa (pointed arch)
    left = arc_pts(ARCH_E, ARCH_YC, r, math.pi + ARCH_ALPHA, math.pi - phi, n)
    right = arc_pts(-ARCH_E, ARCH_YC, r, -ARCH_ALPHA, phi, n)
    return left, right


def jamb_x(offset=0.0):
    return -ARCH_E + (ARCH_R + offset) * math.cos(ARCH_ALPHA)


def spring_y(offset=0.0):
    return ARCH_YC - (ARCH_R + offset) * math.sin(ARCH_ALPHA)


def niche_outline():
    left, right = arch_halves()
    w = jamb_x()
    pts = [(-w, BASE_Y)] + left + right[::-1] + [(w, BASE_Y)]
    return pts


def voussoirs(count_per_side=9):
    """Wedge stones of the arch, as polygons, ordered springer -> keystone."""
    inner_l, inner_r = arch_halves(0.0, 400)
    outer_l, outer_r = arch_halves(VOUSSOIR, 400)
    stones = []
    for side, (inn, out) in enumerate(((inner_l, outer_l), (inner_r, outer_r))):
        n = len(inn)
        for k in range(count_per_side):
            a = int(k * (n - 1) / count_per_side)
            b = int((k + 1) * (n - 1) / count_per_side)
            poly = inn[a:b + 1] + out[a:b + 1][::-1]
            stones.append((k, side, poly))
    return stones


def polyline_length(pts):
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))


def partial_polyline(pts, frac):
    """First `frac` of a polyline, by arc length."""
    if frac >= 1:
        return pts
    if frac <= 0:
        return pts[:1]
    target = polyline_length(pts) * frac
    out, acc = [pts[0]], 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if acc + seg >= target:
            k = (target - acc) / seg if seg else 0
            out.append((a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k))
            return out
        out.append(b)
        acc += seg
    return out


# ----------------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------------
TILES = build_pattern(W * 0.9, H * 0.75)
NICHE = niche_outline()
ARCH_L, ARCH_R_PTS = arch_halves(0.0, 120)
ARCH_OUT_L, ARCH_OUT_R = arch_halves(VOUSSOIR, 120)
STONES = voussoirs()
JAMB = jamb_x()
OUTER_JAMB = jamb_x(VOUSSOIR)
APEX_OUT = ARCH_OUT_L[-1][1]
ALFIZ = (-OUTER_JAMB - ALFIZ_PAD, spring_y(VOUSSOIR) - 40,
         OUTER_JAMB + ALFIZ_PAD, APEX_OUT + ALFIZ_PAD)  # x0, y0, x1, y1
HOOD_Y = spring_y() + 70     # bottom edge of the shell hood

rng = np.random.default_rng(3)
DUST = rng.random((140, 4))  # x, y, speed, size


def camera(t):
    """Return (zoom, cx, cy) in world coordinates."""
    # tight on the first rosette, pull out, settle on the mihrab, slow push.
    z0 = 4.2
    k1 = ease_io_cubic(prog(t, 2.6, 8.0))
    zoom = z0 + (1.12 - z0) * k1
    cy = 0.0 + (-40.0) * k1
    k2 = ease_io(prog(t, 8.0, 12.0))
    zoom = zoom + (0.98 - 1.12) * k2
    cy = cy + (-80.0 - -40.0) * k2
    k3 = ease_io(prog(t, 12.0, 20.0))
    zoom = zoom + 0.07 * k3
    cy = cy + 25.0 * k3
    return zoom, 0.0, cy


def path_poly(ctx, pts, close=True):
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    if close:
        ctx.close_path()


def draw_background(ctx, t):
    g = cairo.RadialGradient(W / 2, H * 0.42, 50, W / 2, H * 0.45, H * 0.75)
    g.add_color_stop_rgb(0, *NIGHT2)
    g.add_color_stop_rgb(1, *NIGHT)
    ctx.set_source(g)
    ctx.paint()


def tile_progress(tile, t):
    """Strapwork drawing progress of a tile, rippling outward from the centre."""
    start = 2.2 + tile.dist / 260.0 + tile.seed * 0.15
    return ease_out(prog(t, start, start + 0.9))


def tile_fill(tile, t):
    start = 6.6 + tile.dist / 420.0 + tile.seed * 0.25
    return ease_out(prog(t, start, start + 0.6))


def draw_pattern(ctx, t, zoom):
    # --- zellige fills -------------------------------------------------------
    for tile in TILES:
        f = tile_fill(tile, t)
        if f <= 0:
            continue
        cx, cy = tile.center
        ctx.save()
        ctx.translate(cx, cy)
        s = 0.6 + 0.4 * f
        ctx.scale(s, s)
        ctx.translate(-cx, -cy)
        # corner pieces (the background "field" between stars)
        n = len(tile.poly)
        field = GREEN if tile.kind == "oct" else COBALT
        for i in range(n):
            m_i, x_i = tile.star[2 * i], tile.star[2 * i + 1]
            m_j = tile.star[(2 * i + 2) % (2 * n)]
            v = tile.poly[(i + 1) % n]
            path_poly(ctx, [m_i, v, m_j, x_i])
        ctx.set_source_rgba(*field, f)
        ctx.fill()
        # the star itself
        path_poly(ctx, tile.star)
        col = CREAM if tile.kind == "oct" else HONEY
        ctx.set_source_rgba(*col, f)
        ctx.fill()
        # little inner rosette on the big stars
        if tile.kind == "oct":
            inner = regular_polygon(cx, cy, 8, EDGE * 0.42, math.pi / 8)
            path_poly(ctx, inner)
            ctx.set_source_rgba(*TERRA, f)
            ctx.fill()
        ctx.restore()

    # --- strapwork lines (gold band with dark core) -------------------------
    segs = []
    for tile in TILES:
        p = tile_progress(tile, t)
        if p <= 0:
            continue
        st = tile.star
        n = len(st)
        for i in range(0, n, 2):
            m, x, m2 = st[i], st[i + 1], st[(i + 2) % n]
            # both arms grow from the edge midpoints towards the crossing X
            segs.append((m, (m[0] + (x[0] - m[0]) * p, m[1] + (x[1] - m[1]) * p)))
            segs.append((m2, (m2[0] + (x[0] - m2[0]) * p, m2[1] + (x[1] - m2[1]) * p)))
    if not segs:
        return
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    for a, b in segs:
        ctx.move_to(*a)
        ctx.line_to(*b)
    # moving shimmer across the gold
    sh = (t * 0.22) % 1.6 - 0.3
    g = cairo.LinearGradient(-W, -H, W, H)
    g.add_color_stop_rgb(0, *GOLD)
    g.add_color_stop_rgb(clamp(sh - 0.08), *GOLD)
    g.add_color_stop_rgb(clamp(sh), *GOLD_HI)
    g.add_color_stop_rgb(clamp(sh + 0.08), *GOLD)
    g.add_color_stop_rgb(1, *GOLD)
    ctx.set_source(g)
    ctx.set_line_width(6.2)
    ctx.stroke_preserve()
    ctx.set_source_rgb(*INK)
    ctx.set_line_width(2.0)
    ctx.stroke()


def draw_construction(ctx, t, zoom):
    """Compass & straightedge prologue on the central octagon."""
    a = 1 - prog(t, 4.0, 5.5)
    if a <= 0:
        return
    lw = 1.4 / zoom * 1.6
    ctx.set_line_width(lw)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    R = EDGE / (2 * math.sin(math.pi / 8))
    # the point
    dot = ease_out(prog(t, 0.3, 0.9))
    if dot > 0:
        g = cairo.RadialGradient(0, 0, 0, 0, 0, 18 * dot)
        g.add_color_stop_rgba(0, *GOLD_HI, 0.9 * a)
        g.add_color_stop_rgba(1, *GOLD, 0)
        ctx.set_source(g)
        ctx.arc(0, 0, 18 * dot, 0, 2 * math.pi)
        ctx.fill()
        ctx.set_source_rgba(*GOLD_HI, a)
        ctx.arc(0, 0, 1.6 * dot, 0, 2 * math.pi)
        ctx.fill()
    # circle
    c = ease_io(prog(t, 0.8, 1.7))
    if c > 0:
        ctx.set_source_rgba(*CREAM, 0.55 * a)
        ctx.arc(0, 0, R, -math.pi / 2, -math.pi / 2 + 2 * math.pi * c)
        ctx.stroke()
    # 8 radii / diameters
    d = prog(t, 1.4, 2.2)
    for k in range(4):
        dk = ease_out(clamp(d * 4 - k))
        if dk <= 0:
            continue
        ang = math.pi / 8 + k * math.pi / 4
        ex, ey = math.cos(ang) * R * 1.25 * dk, math.sin(ang) * R * 1.25 * dk
        ctx.move_to(-ex, -ey)
        ctx.line_to(ex, ey)
        ctx.set_source_rgba(*CREAM, 0.35 * a)
        ctx.stroke()
    # octagon
    o = ease_io(prog(t, 1.9, 2.6))
    if o > 0:
        oct_ = regular_polygon(0, 0, 8, R, math.pi / 8)
        path_poly(ctx, partial_polyline(oct_ + [oct_[0]], o), close=False)
        ctx.set_source_rgba(*CREAM, 0.6 * a)
        ctx.stroke()


def draw_mihrab(ctx, t, zoom):
    reveal = ease_io(prog(t, 8.6, 10.0))
    if reveal <= 0:
        return

    # 1. darken everything outside the niche so the arch reads as an opening
    ctx.save()
    ctx.rectangle(-W * 3, -H * 3, W * 6, H * 6)
    path_poly(ctx, NICHE)
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.clip()
    ctx.set_source_rgba(*NIGHT, 0.78 * reveal)
    ctx.paint()
    ctx.restore()

    # 2. alfiz frame (rectangle around the arch) + spandrels
    x0, y0, x1, y1 = ALFIZ
    fa = ease_out(prog(t, 10.0, 11.4))
    if fa > 0:
        # spandrel fill: deep green stucco with fine gold lattice
        ctx.save()
        ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
        outer = ARCH_OUT_L + ARCH_OUT_R[::-1]
        path_poly(ctx, outer)
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.clip()
        ctx.set_source_rgba(0.04, 0.20, 0.18, 0.92 * fa)
        ctx.paint()
        ctx.set_source_rgba(*GOLD, 0.35 * fa)
        ctx.set_line_width(1.2)
        step = 22
        for k in range(-40, 40):
            ctx.move_to(x0 + k * step, y0)
            ctx.line_to(x0 + k * step + (y1 - y0), y1)
            ctx.move_to(x0 + k * step, y1)
            ctx.line_to(x0 + k * step + (y1 - y0), y0)
        ctx.stroke()
        # small 8-point stars in the spandrels
        for sx in (x0 + 72, x1 - 72):
            star = hankin(regular_polygon(sx, y1 - 78, 8, 40, math.pi / 8), THETA)
            path_poly(ctx, star)
            ctx.set_source_rgba(*GOLD, fa)
            ctx.fill()
        ctx.restore()

        # frame lines draw themselves
        frame = [(0, y1), (x0, y1), (x0, y0)]
        frame_r = [(0, y1), (x1, y1), (x1, y0)]
        ctx.set_line_width(5)
        ctx.set_source_rgba(*GOLD, 1)
        for fr in (frame, frame_r):
            path_poly(ctx, partial_polyline(fr, fa), close=False)
            ctx.stroke()
        # epigraphic band above the alfiz: a frieze of little stars
        band_y = y1 + 46
        ctx.set_line_width(3)
        bx = partial_polyline([(x0, band_y + 34), (x1, band_y + 34)], fa)
        path_poly(ctx, bx, close=False)
        ctx.stroke()
        bx = partial_polyline([(x0, band_y - 34), (x1, band_y - 34)], fa)
        path_poly(ctx, bx, close=False)
        ctx.stroke()
        n_st = 9
        for k in range(n_st):
            sx = x0 + (k + 0.5) * (x1 - x0) / n_st
            sk = ease_out(prog(t, 10.6 + k * 0.07, 11.1 + k * 0.07))
            if sk <= 0:
                continue
            star = hankin(regular_polygon(sx, band_y, 8, 24 * sk, math.pi / 8), THETA)
            path_poly(ctx, star)
            ctx.set_source_rgba(*GOLD, sk)
            ctx.fill()

    # 3. voussoirs: alternating terracotta / cream stones, springer -> keystone
    for k, side, poly in STONES:
        sk = ease_out(prog(t, 9.6 + k * 0.11, 10.1 + k * 0.11))
        if sk <= 0:
            continue
        col = TERRA if k % 2 == 0 else CREAM
        path_poly(ctx, poly)
        ctx.set_source_rgba(*col, sk)
        ctx.fill_preserve()
        ctx.set_source_rgba(*GOLD, sk)
        ctx.set_line_width(2.2)
        ctx.stroke()
        # a small rosette on each stone
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        ctx.arc(cx, cy, 6 * sk, 0, 2 * math.pi)
        ctx.set_source_rgba(*(GOLD if k % 2 == 0 else TERRA), sk)
        ctx.fill()

    # 4. shell hood inside the arch head
    hd = prog(t, 11.2, 13.0)
    if hd > 0:
        ctx.save()
        path_poly(ctx, NICHE)
        ctx.clip()
        ctx.rectangle(-W, HOOD_Y, 2 * W, H)
        ctx.clip()
        n_ribs = 15
        apex = (0, HOOD_Y)
        for k in range(n_ribs):
            # wedges unfold from both sides toward the centre
            order = min(k, n_ribs - 1 - k)
            wk = ease_out(clamp(hd * 2.2 - order * 0.18))
            if wk <= 0:
                continue
            a0 = math.pi * k / n_ribs
            a1 = math.pi * (k + 1) / n_ribs
            Rr = 600
            ctx.move_to(*apex)
            ctx.line_to(apex[0] + Rr * math.cos(a0), apex[1] + Rr * math.sin(a0))
            ctx.line_to(apex[0] + Rr * math.cos(a1), apex[1] + Rr * math.sin(a1))
            ctx.close_path()
            col = (0.90, 0.85, 0.74) if k % 2 == 0 else (0.80, 0.64, 0.38)
            ctx.set_source_rgba(*col, wk)
            ctx.fill()
            ctx.move_to(*apex)
            ctx.line_to(apex[0] + Rr * math.cos(a0), apex[1] + Rr * math.sin(a0))
            ctx.set_source_rgba(*GOLD, wk)
            ctx.set_line_width(2.4)
            ctx.stroke()
        # concave shading: the shell gets darker towards its rim
        g = cairo.RadialGradient(apex[0], apex[1], 20, apex[0], apex[1], 420)
        g.add_color_stop_rgba(0, 1, 0.95, 0.8, 0.25 * hd)
        g.add_color_stop_rgba(0.45, 0, 0, 0, 0.0)
        g.add_color_stop_rgba(1, 0.05, 0.03, 0.02, 0.55 * hd)
        ctx.set_source(g)
        ctx.paint()
        ctx.restore()
        # shadow cast by the hood onto the tiles below
        ctx.save()
        path_poly(ctx, NICHE)
        ctx.clip()
        g = cairo.LinearGradient(0, HOOD_Y, 0, HOOD_Y - 160)
        g.add_color_stop_rgba(0, 0, 0, 0, 0.55 * ease_out(hd))
        g.add_color_stop_rgba(1, 0, 0, 0, 0)
        ctx.rectangle(-W, HOOD_Y - 160, 2 * W, 160)
        ctx.set_source(g)
        ctx.fill()
        ctx.restore()
        # scalloped cornice under the hood
        n_lobes = 9
        lw_ = 2 * JAMB / n_lobes
        for k in range(n_lobes):
            lk = ease_out(clamp(hd * 2 - abs(k - n_lobes // 2) * 0.12))
            if lk <= 0:
                continue
            cx = -JAMB + (k + 0.5) * lw_
            ctx.new_sub_path()
            ctx.arc_negative(cx, HOOD_Y, lw_ / 2 * lk, 0, math.pi)
            ctx.set_source_rgba(*GOLD, lk)
            ctx.set_line_width(4)
            ctx.stroke()
        ctx.move_to(-JAMB, HOOD_Y)
        ctx.line_to(-JAMB + 2 * JAMB * ease_out(clamp(hd * 2)), HOOD_Y)
        ctx.set_line_width(5)
        ctx.stroke()

    # 5. colonnettes rising on both jambs
    col_k = ease_out(prog(t, 9.0, 10.4))
    if col_k > 0:
        cw = 26
        top = spring_y()
        for sgn in (-1, 1):
            xc = sgn * (JAMB + cw / 2 - 4)
            hgt = (top - BASE_Y) * col_k
            g = cairo.LinearGradient(xc - cw / 2, 0, xc + cw / 2, 0)
            g.add_color_stop_rgb(0, 0.55, 0.42, 0.20)
            g.add_color_stop_rgb(0.45, *GOLD_HI)
            g.add_color_stop_rgb(1, 0.45, 0.33, 0.15)
            ctx.rectangle(xc - cw / 2, BASE_Y, cw, hgt)
            ctx.set_source(g)
            ctx.fill()
            # capital
            if col_k > 0.95:
                ctx.rectangle(xc - cw * 0.9, top - 4, cw * 1.8, 26)
                ctx.set_source(g)
                ctx.fill()
            # base
            ctx.rectangle(xc - cw * 0.85, BASE_Y, cw * 1.7, 22)
            ctx.set_source(g)
            ctx.fill()

    # 6. the arch outline draws itself from the floor up to the apex
    ak = ease_io(prog(t, 8.8, 10.2))
    if ak > 0:
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for half, sgn in ((ARCH_L, -1), (ARCH_R_PTS, 1)):
            pts = [(sgn * JAMB, BASE_Y)] + half
            path_poly(ctx, partial_polyline(pts, ak), close=False)
            ctx.set_source_rgba(*GOLD_HI, 1)
            ctx.set_line_width(5.5)
            ctx.stroke()
        for half in (ARCH_OUT_L, ARCH_OUT_R):
            path_poly(ctx, partial_polyline(half, ak), close=False)
            ctx.set_source_rgba(*GOLD, 1)
            ctx.set_line_width(4)
            ctx.stroke()

    # 7. dado / floor line
    fl = ease_out(prog(t, 9.2, 10.4))
    if fl > 0:
        ctx.move_to(-x1 * fl, BASE_Y)
        ctx.line_to(x1 * fl, BASE_Y)
        ctx.set_source_rgba(*GOLD, 1)
        ctx.set_line_width(5)
        ctx.stroke()


def draw_light(ctx, t):
    """Screen-space light: god ray from the top, dust motes, vignette."""
    lk = ease_io(prog(t, 12.0, 14.5))
    if lk > 0:
        ctx.save()
        ctx.set_operator(cairo.OPERATOR_ADD)
        sway = math.sin(t * 0.6) * 30
        ctx.move_to(W * 0.56 + sway, -50)
        ctx.line_to(W * 0.86 + sway, -50)
        ctx.line_to(W * 0.62, H * 0.95)
        ctx.line_to(W * 0.18, H * 0.95)
        ctx.close_path()
        g = cairo.LinearGradient(0, 0, 0, H)
        g.add_color_stop_rgba(0, 1.0, 0.85, 0.55, 0.22 * lk)
        g.add_color_stop_rgba(0.6, 1.0, 0.80, 0.50, 0.08 * lk)
        g.add_color_stop_rgba(1, 1.0, 0.80, 0.50, 0.0)
        ctx.set_source(g)
        ctx.fill()
        # dust
        for x, y, sp, sz in DUST:
            yy = (y * H * 1.1 - t * (8 + 22 * sp)) % (H * 1.1) - 40
            xx = x * W + math.sin(t * (0.5 + sp) + y * 9) * 20
            # only inside the beam (roughly)
            frac = yy / H
            left = (W * 0.56 + sway) + (W * 0.18 - (W * 0.56 + sway)) * frac
            right = (W * 0.86 + sway) + (W * 0.62 - (W * 0.86 + sway)) * frac
            inside = 1.0 if left < xx < right else 0.25
            tw = 0.5 + 0.5 * math.sin(t * 3 + x * 40)
            ctx.arc(xx, yy, 1.0 + 2.2 * sz, 0, 2 * math.pi)
            ctx.set_source_rgba(1, 0.9, 0.7, 0.55 * lk * inside * tw)
            ctx.fill()
        ctx.restore()

    # vignette
    g = cairo.RadialGradient(W / 2, H / 2, H * 0.30, W / 2, H / 2, H * 0.72)
    g.add_color_stop_rgba(0, 0, 0, 0, 0)
    g.add_color_stop_rgba(1, 0, 0, 0, 0.75)
    ctx.set_source(g)
    ctx.paint()


def draw_text(ctx, t):
    tk = prog(t, 14.6, 16.4)
    if tk <= 0:
        return
    ctx.save()
    # title
    title = "MIHRAB"
    ctx.select_font_face("Caladea", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(112)
    spacing = 46 - 22 * ease_out(tk)
    widths = [ctx.text_extents(ch).x_advance for ch in title]
    total = sum(widths) + spacing * (len(title) - 1)
    x = (W - total) / 2
    y = H * 0.885
    for i, ch in enumerate(title):
        ck = ease_out(clamp(tk * 1.6 - i * 0.09))
        ctx.move_to(x, y + 26 * (1 - ck))
        ctx.text_path(ch)
        ctx.set_source_rgba(*GOLD_HI, ck)
        ctx.fill()
        x += widths[i] + spacing
    # subtitle in monospace, typed out like code
    sub = "drawn entirely with code"
    sk = prog(t, 15.8, 17.6)
    n = int(len(sub) * sk)
    if n > 0:
        ctx.select_font_face("JetBrains Mono", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(34)
        full = ctx.text_extents(sub).x_advance
        shown = sub[:n]
        cursor = "_" if (int(t * 3) % 2 == 0 or n < len(sub)) else " "
        ctx.move_to((W - full) / 2, y + 74)
        ctx.set_source_rgba(*CREAM, 0.85)
        ctx.show_text(shown + cursor)
    hk = ease_out(prog(t, 17.4, 18.4))
    if hk > 0:
        ctx.select_font_face("JetBrains Mono", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(28)
        handle = "@0skndr"
        ex = ctx.text_extents(handle).x_advance
        ctx.move_to((W - ex) / 2, y + 128)
        ctx.set_source_rgba(*GOLD, hk * 0.9)
        ctx.show_text(handle)
    ctx.restore()


def render_frame(t):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    draw_background(ctx, t)

    zoom, cx, cy = camera(t)
    ctx.save()
    ctx.translate(W / 2, H / 2)
    ctx.scale(zoom, -zoom)          # world is y-up
    ctx.translate(-cx, -cy)
    draw_construction(ctx, t, zoom)
    draw_pattern(ctx, t, zoom)
    draw_mihrab(ctx, t, zoom)
    ctx.restore()

    draw_light(ctx, t)
    draw_text(ctx, t)

    # fade in / out
    fade = min(prog(t, 0, 0.4), 1 - prog(t, DURATION - 0.6, DURATION))
    if fade < 1:
        ctx.set_source_rgba(0, 0, 0, 1 - fade)
        ctx.paint()
    surf.flush()
    return surf


def frame_bytes(i):
    surf = render_frame(i / FPS)
    return bytes(surf.get_data())  # BGRA


# ----------------------------------------------------------------------------
# Audio (all synthesized)
# ----------------------------------------------------------------------------
SR = 44100


def karplus(freq, dur, bright=0.5, rng=None):
    n = int(SR * dur)
    period = int(SR / freq)
    buf = rng.uniform(-1, 1, period)
    out = np.zeros(n)
    # vectorised-ish Karplus-Strong: process one period at a time
    pos = 0
    while pos < n:
        m = min(period, n - pos)
        out[pos:pos + m] = buf[:m]
        nxt = 0.996 * (bright * buf + (1 - bright) * np.roll(buf, 1) * 0.5 + (1 - bright) * buf * 0.5)
        buf = 0.5 * (nxt + np.roll(nxt, -1))
        pos += m
    env = np.minimum(1, np.arange(n) / (SR * 0.004))
    return out * env


def reverb(x, seconds=3.2, mix_=0.35, seed=1):
    r = np.random.default_rng(seed)
    n = int(SR * seconds)
    ir = r.standard_normal(n) * np.exp(-np.linspace(0, 7, n))
    ir[0] = 0
    ir /= np.sqrt(np.sum(ir ** 2))
    L = len(x) + n
    nfft = 1 << (L - 1).bit_length()
    wet = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[:len(x)]
    return (1 - mix_) * x + mix_ * wet


def build_audio(path):
    rng = np.random.default_rng(11)
    n = int(SR * DURATION)
    t = np.arange(n) / SR
    L = np.zeros(n)
    Rch = np.zeros(n)

    D = 146.83  # D3
    # Maqam Hijaz on D: D Eb F# G A Bb C D
    hijaz = [0, 1, 4, 5, 7, 8, 10, 12]

    def note(step, octave=0):
        semis = hijaz[step % 7] + 12 * (step // 7 + octave)
        return D * 2 ** (semis / 12)

    # drone: D + A, slowly breathing, swells as the mihrab appears
    swell = np.clip((t - 0.3) / 3, 0, 1) * (0.55 + 0.45 * np.clip((t - 8.5) / 3, 0, 1))
    swell *= 1 - np.clip((t - (DURATION - 1.5)) / 1.5, 0, 1)
    drone = (np.sin(2 * np.pi * D / 2 * t) * 0.5 + np.sin(2 * np.pi * D * 0.75 * t + 0.3) * 0.25
             + np.sin(2 * np.pi * D * t * 1.002) * 0.18 + np.sin(2 * np.pi * D * 1.5 * t) * 0.08)
    drone *= (0.8 + 0.2 * np.sin(2 * np.pi * 0.11 * t)) * swell * 0.16
    L += drone
    Rch += drone * 0.95

    # plucks: a phrase that follows the picture
    events = [(0.35, 0, 0), (0.85, 4, 0), (1.45, 2, 0), (1.95, 4, 0), (2.35, 7, 0)]
    # ripple of the pattern: rising arpeggio
    for k, st in enumerate([0, 2, 3, 4, 5, 4, 6, 7, 6, 5, 4, 3, 4, 2, 1, 0]):
        events.append((2.7 + k * 0.32, st, 1 if k % 4 == 0 else 0))
    # zellige pops: soft high notes
    for k in range(10):
        events.append((6.8 + k * 0.18, [7, 9, 11, 8, 10, 12, 11, 9, 8, 7][k], 0))
    # mihrab reveal: low strong notes
    events += [(8.7, 0, -1), (9.6, 4, -1), (10.4, 3, -1), (11.2, 0, 0),
               (12.0, 4, 0), (12.6, 5, 0), (13.2, 4, 0), (13.8, 1, 0), (14.6, 0, 0),
               (15.6, 7, 0), (16.6, 4, 0), (17.6, 0, 0)]
    for (when, step, octv) in events:
        f = note(step, octv)
        dur = 2.6
        p = karplus(f, dur, bright=0.55 if octv >= 0 else 0.4, rng=rng)
        amp = 0.32 if octv < 0 else 0.22
        i0 = int(when * SR)
        i1 = min(n, i0 + len(p))
        pan = 0.5 + 0.35 * math.sin(step * 1.7)
        L[i0:i1] += p[:i1 - i0] * amp * (1 - pan) * 2
        Rch[i0:i1] += p[:i1 - i0] * amp * pan * 2

    # a soft "whoosh" when the camera pulls back, and a shimmer when the light enters
    noise = rng.standard_normal(n)
    sm = np.convolve(noise, np.ones(40) / 40, mode="same")
    wh = np.exp(-((t - 5.0) / 1.6) ** 2) * 0.05
    L += sm * wh
    Rch += np.roll(sm, 300) * wh
    bell = sum(np.sin(2 * np.pi * note(s, 2) * t) * 0.03 for s in (0, 4, 7))
    bell *= np.exp(-np.maximum(0, t - 12.2) * 0.8) * (t > 12.2)
    L += bell
    Rch += bell

    L = reverb(L, seed=1)
    Rch = reverb(Rch, seed=2)
    stereo = np.stack([L, Rch], axis=1)
    stereo /= np.max(np.abs(stereo)) * 1.12
    pcm = (stereo * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", type=float, help="render a single frame at this time (s)")
    ap.add_argument("--out", default=os.path.join(HERE, "mihrab.mp4"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    if args.still is not None:
        surf = render_frame(args.still)
        path = os.path.join(HERE, f"still_{args.still:05.2f}.png")
        surf.write_to_png(path)
        print(path)
        return

    wav = os.path.join(HERE, "mihrab_audio.wav")
    print("synthesizing audio ...")
    build_audio(wav)

    total = int(DURATION * FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", wav,
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
           args.out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(args.workers) as pool:
        for i, data in enumerate(pool.imap(frame_bytes, range(total), chunksize=4)):
            ff.stdin.write(data)
            if i % 30 == 0:
                print(f"frame {i}/{total}", file=sys.stderr)
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

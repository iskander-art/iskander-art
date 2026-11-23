"""Soundtrack for the mihrab reel, synthesized from scratch with numpy.

A warm drone, a slow oud-like melody in maqam Hijaz (Karplus-Strong plucks),
soft frame-drum hits on every cut and a shimmer when the wide shot opens.
Usage: python audio.py out.wav
"""
import math
import sys
import wave

import numpy as np

SR = 44100
DURATION = 20.0
CUTS = [3.6, 7.2, 10.6, 15.8]
rng = np.random.default_rng(5)
n = int(SR * DURATION)
t = np.arange(n) / SR

D = 146.83                              # D3
HIJAZ = [0, 1, 4, 5, 7, 8, 10]          # D Eb F# G A Bb C


def hz(step, octave=0):
    return D * 2 ** ((HIJAZ[step % 7] + 12 * (step // 7 + octave)) / 12)


def pluck(freq, dur, damp=0.996, bright=0.5):
    N = int(SR * dur)
    period = max(2, int(round(SR / freq)))
    buf = rng.uniform(-1, 1, period)
    buf = np.convolve(buf, [bright, 1 - bright], mode="same")
    out = np.empty(N)
    pos = 0
    while pos < N:
        m = min(period, N - pos)
        out[pos:pos + m] = buf[:m]
        buf = damp * 0.5 * (buf + np.roll(buf, -1))
        pos += m
    return out * np.minimum(1, np.arange(N) / (SR * 0.003))


def daf(dur=1.2):
    N = int(SR * dur)
    tt = np.arange(N) / SR
    body = np.sin(2 * np.pi * 70 * tt * (1 - 0.25 * tt)) * np.exp(-tt * 9)
    skin = rng.standard_normal(N) * np.exp(-tt * 30) * 0.35
    skin = np.convolve(skin, np.ones(6) / 6, mode="same")
    jingles = rng.standard_normal(N) * np.exp(-tt * 14) * 0.12
    jingles = np.diff(jingles, prepend=0) * 3
    return body + skin + jingles


def reverb(x, seconds=3.5, wet=0.4, seed=1):
    r = np.random.default_rng(seed)
    N = int(SR * seconds)
    ir = r.standard_normal(N) * np.exp(-np.linspace(0, 6.5, N))
    ir[: int(SR * 0.012)] = 0
    ir /= np.sqrt(np.sum(ir ** 2))
    L = len(x) + N
    nfft = 1 << (L - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[: len(x)]
    return (1 - wet) * x + wet * y


L = np.zeros(n)
R = np.zeros(n)


def place(sig, when, amp, pan=0.5):
    i0 = int(when * SR)
    i1 = min(n, i0 + len(sig))
    if i0 >= n:
        return
    L[i0:i1] += sig[: i1 - i0] * amp * math.sqrt(1 - pan)
    R[i0:i1] += sig[: i1 - i0] * amp * math.sqrt(pan)


# drone (D + A + soft octave), breathing in and out
env = np.clip(t / 2.5, 0, 1) * (1 - np.clip((t - (DURATION - 2.0)) / 2.0, 0, 1))
drone = (0.55 * np.sin(2 * np.pi * D / 2 * t) + 0.25 * np.sin(2 * np.pi * D * 0.75 * t + 1)
         + 0.15 * np.sin(2 * np.pi * D * 1.003 * t) + 0.06 * np.sin(2 * np.pi * D * 1.5 * t))
drone *= env * (0.85 + 0.15 * np.sin(2 * np.pi * 0.13 * t)) * 0.13
L += drone
R += drone

# melody: (time, step, octave, length)
melody = [
    (0.4, 0, 0), (1.3, 1, 0), (1.9, 2, 0), (2.6, 3, 0), (3.1, 2, 0),
    (3.6, 4, 0), (4.5, 3, 0), (5.0, 2, 0), (5.6, 3, 0), (6.3, 1, 0), (6.8, 0, 0),
    (7.2, 4, 0), (7.8, 5, 0), (8.3, 4, 0), (8.9, 6, 0), (9.5, 5, 0), (10.0, 4, 0),
    (10.6, 7, 0), (11.6, 6, 0), (12.2, 5, 0), (12.9, 4, 0), (13.6, 3, 0), (14.3, 2, 0), (15.0, 1, 0),
    (15.8, 0, 0), (16.8, 2, 0), (17.5, 1, 0), (18.4, 0, 0),
]
for when, step, octv in melody:
    f = hz(step, octv)
    place(pluck(f, 3.0, damp=0.997, bright=0.6), when, 0.30, 0.42 + 0.16 * math.sin(step))
    place(pluck(f * 2, 1.2, damp=0.993, bright=0.7), when + 0.012, 0.06, 0.6)   # string shimmer
# low notes on the cuts
for c in [0.4] + CUTS:
    place(pluck(hz(0, -1), 3.5, damp=0.998, bright=0.35), c, 0.32, 0.5)
# frame drum on every cut, plus a soft pulse
for c in CUTS:
    place(daf(), c, 0.55, 0.5)
for k in range(int(DURATION / 1.8)):
    tt = 1.8 * k + 0.4
    if all(abs(tt - c) > 0.3 for c in CUTS) and tt < DURATION - 1.5:
        place(daf(0.6), tt, 0.13, 0.35 + 0.3 * (k % 2))

# shimmer when the wide shot opens
for k, step in enumerate([7, 9, 11, 14, 11, 9]):
    place(np.sin(2 * np.pi * hz(step, 1) * np.arange(int(SR * 2.5)) / SR) * np.exp(-np.arange(int(SR * 2.5)) / SR * 2.2),
          10.6 + k * 0.09, 0.035, 0.2 + 0.12 * k)

L = reverb(L, seed=1)
R = reverb(R, seed=2)
st = np.stack([L, R], 1)
fade = np.ones(n)
fade[-int(SR * 1.2):] = np.linspace(1, 0, int(SR * 1.2))
st *= fade[:, None]
st /= np.max(np.abs(st)) * 1.1
out = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
with wave.open(out, "wb") as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(SR)
    wf.writeframes((st * 32767).astype(np.int16).tobytes())
print("audio ->", out)

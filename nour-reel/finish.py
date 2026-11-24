"""
Sound design + mix + final composite for the Nour Gallery reel.

  python finish.py --voice voice.mp3 --picture picture.mp4 --captions captions.mov --out reel.mp4

Synthesizes the sound effects with numpy (low boom at the start, whooshes on the cuts,
soft pops on highlighted captions, impact on the brand flash, shimmer on the end card),
processes the voice, mixes, loudness-normalizes to -14 LUFS and overlays the captions.
"""
import argparse
import json
import os
import subprocess
import wave

import numpy as np

import reel

SR = 48000
rng = np.random.default_rng(21)


def boom(dur=1.6):
    t = np.arange(int(SR * dur)) / SR
    f = 55 * np.exp(-t * 1.2) + 30
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.2)
    click = rng.standard_normal(len(t)) * np.exp(-t * 60) * 0.3
    return body + np.convolve(click, np.ones(30) / 30, mode="same")


def whoosh(dur=0.7, peak=0.5, bright=1.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    env = np.exp(-((t - dur * peak) / (dur * 0.22)) ** 2)
    noise = rng.standard_normal(n)
    # sweep: blend from a dark to a brighter filtered version around the peak
    dark = np.convolve(noise, np.ones(48) / 48, mode="same")
    mid = np.convolve(noise, np.ones(10) / 10, mode="same")
    mix = dark * (1 - env * bright) + mid * env * bright
    return mix * env * 2.2


def pop(dur=0.09):
    t = np.arange(int(SR * dur)) / SR
    f = 600 + 900 * t / dur
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / dur) ** 2 * 0.5


def shimmer(dur=1.6):
    t = np.arange(int(SR * dur)) / SR
    out = np.zeros_like(t)
    for k, f in enumerate([1318.5, 1567.98, 1975.5, 2349.3, 2637.0]):
        start = k * 0.06
        tt = np.clip(t - start, 0, None)
        out += np.sin(2 * np.pi * f * tt) * np.exp(-tt * 3.0) * (t >= start)
    return out * 0.12


def place(buf, sig, at, gain):
    i = int(at * SR)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[: j - i] * gain


def write_wav(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--picture", required=True)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work = os.path.dirname(os.path.abspath(a.out))
    dur = reel.DURATION

    sfx = np.zeros(int(SR * (dur + 1)))
    place(sfx, boom(), 0.0, 0.55)
    for c in reel.CUTS:
        if abs(c - reel.FLASH_AT) < 0.01:
            continue
        place(sfx, whoosh(0.7, 0.55), c - 0.385, 0.22)
    pops = json.load(open(a.captions + ".pops.json"))
    for p in pops:
        place(sfx, pop(), p, 0.20)
    place(sfx, whoosh(1.0, 0.6, 1.2), reel.FLASH_AT - 0.6, 0.35)          # bigger whoosh into the flash
    place(sfx, boom(1.2), reel.FLASH_AT, 0.40)                              # impact
    end_at = 26.96
    place(sfx, whoosh(0.8, 0.5), end_at - 0.4, 0.25)
    place(sfx, shimmer(), end_at + 0.05, 0.9)
    place(sfx, boom(1.4), end_at + 0.05, 0.30)
    sfx_path = os.path.join(work, "sfx.wav")
    write_wav(sfx_path, sfx[: int(SR * dur)])

    mixed = os.path.join(work, "mix.wav")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", a.voice, "-i", sfx_path,
        "-filter_complex",
        "[0:a]aresample=48000,highpass=f=70,equalizer=f=3500:t=q:w=1:g=2,"
        "acompressor=threshold=-20dB:ratio=3:attack=5:release=120,adelay=0|0,apad=whole_dur=%.2f[v];"
        "[1:a]aresample=48000,volume=1.0[s];"
        "[v][s]amix=inputs=2:normalize=0:duration=first,loudnorm=I=-14:TP=-1.5:LRA=9[out]" % dur,
        "-map", "[out]", "-ar", "48000", "-ac", "2", mixed], check=True)

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", a.picture, "-i", a.captions, "-i", mixed,
        "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto,format=yuv420p[v]",
        "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-c:a", "aac", "-b:a", "192k", "-t", "%.2f" % dur, "-movflags", "+faststart", a.out], check=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

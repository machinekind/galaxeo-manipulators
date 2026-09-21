#!/usr/bin/env python3
"""Re-encode a GIF (from run_pour.py --gif or eval_policy.py --gif) as H.264 mp4.

    sim/.venv-lerobot/bin/python sim/gif2mp4.py pour.gif pour.mp4 [--fps 20] [--crf 23]

PyAV's wheel carries its own libx264, so this needs no ffmpeg binary.
"""
import argparse

import av
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gif")
    ap.add_argument("mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--crf", type=int, default=23)
    a = ap.parse_args()

    im = Image.open(a.gif)
    w, h = im.size
    w, h = w - w % 2, h - h % 2                     # yuv420p needs even sides
    out = av.open(a.mp4, "w")
    stream = out.add_stream("libx264", rate=a.fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.options = {"crf": str(a.crf), "preset": "medium"}
    n = 0
    for k in range(im.n_frames):
        im.seek(k)
        frame = np.asarray(im.convert("RGB"))[:h, :w]
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            out.mux(packet)
        n += 1
    for packet in stream.encode():
        out.mux(packet)
    out.close()
    print(f"wrote {a.mp4}: {n} frames at {a.fps} fps, {w}x{h}")


if __name__ == "__main__":
    main()

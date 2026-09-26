#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["opencv-python-headless", "numpy"]
# ///
"""calib_intrinsics.py with the live view in the browser instead of a window.

    uv run calib_intrinsics_web.py --camera 0 --rotate --cols 6 --rows 4 --square 0.039

Opens http://127.0.0.1:8766: the live frame with the detected corners drawn,
a green banner while the whole board is found, the view counter, and the fit
once enough views are in (the .npz is written the same way calib_intrinsics.py
writes it, so calib_wrist.py and calibrate_real.py read it unchanged).

A view is taken at most once a second while the full board is visible, so
hold the board still for a beat in each new pose. Cover the frame: near and
far, all four corners, tilted every way.
"""
import argparse
import os
import platform
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_intrinsics import CRITERIA, DEFAULT_OUT_DIR, MIN_ERR_PX, fit   # noqa: E402

PAGE = (b'<html><body style="margin:0;background:#000">'
        b'<img src="/mjpg" style="width:100vw;height:100vh;object-fit:contain">'
        b'</body></html>')


class State:
    jpeg = None
    done = False
    text = "starting"


def serve(port):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path != "/mjpg":
                self.send_response(200); self.send_header("Content-Type", "text/html")
                self.end_headers(); self.wfile.write(PAGE); return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    j = State.jpeg
                    if j is not None:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cols", type=int, default=9, help="checkerboard columns (squares)")
    ap.add_argument("--rows", type=int, default=6, help="checkerboard rows (squares)")
    ap.add_argument("--square", type=float, default=0.025, help="square side [m]")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--max-err", type=float, default=MIN_ERR_PX)
    ap.add_argument("--rotate", action="store_true")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(a.camera, backend)
    if not cap.isOpened():
        sys.exit(f"camera {a.camera} did not open")
    for _ in range(10):
        cap.read()
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {a.camera}: {w}x{h}, rotate {'on' if a.rotate else 'off'}")

    serve(a.port)
    url = f"http://127.0.0.1:{a.port}"
    print("live view:", url, flush=True)
    if not a.no_open:
        webbrowser.open(url)

    pattern = (a.cols - 1, a.rows - 1)
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objpoints, imgpoints = [], []
    last, grabbed = 0.0, 0
    result_lines = []
    while True:
        ok, img = cap.read()
        if not ok:
            time.sleep(0.05); continue
        if a.rotate:
            img = cv2.rotate(img, cv2.ROTATE_180)
        vis = img
        if grabbed < a.frames:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            vis = img.copy()
            if found:
                cv2.drawChessboardCorners(vis, pattern, corners, found)
            now = time.time()
            took = False
            if found and now - last > 1.0:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
                objpoints.append(objp); imgpoints.append(corners)
                grabbed += 1; last = now; took = True
                print(f"  view {grabbed:2d}/{a.frames}", flush=True)
            colour = (0, 255, 0) if found else (0, 0, 255)
            cv2.rectangle(vis, (0, 0), (w, 70), colour, -1)
            msg = (f"VIEW {grabbed}/{a.frames} TAKEN" if took else
                   f"board found - hold still  ({grabbed}/{a.frames})" if found else
                   f"show the WHOLE board  ({grabbed}/{a.frames})")
            cv2.putText(vis, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
            if grabbed >= a.frames:
                rms, K, dist = fit(objpoints, imgpoints, (w, h))
                out = a.out or os.path.join(DEFAULT_OUT_DIR, "K.npz")
                os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                np.savez(out, K=K, dist=dist, rms=rms, views=grabbed, cols=a.cols, rows=a.rows,
                         square=a.square, rotate=bool(a.rotate), width=w, height=h,
                         trusted=bool(rms <= a.max_err),
                         objpoints=np.asarray(objpoints), imgpoints=np.asarray(imgpoints))
                fovx = np.degrees(2 * np.arctan(w / (2 * K[0, 0])))
                fovy = np.degrees(2 * np.arctan(h / (2 * K[1, 1])))
                result_lines = [f"DONE rms {rms:.2f} px {'OK' if rms <= a.max_err else 'UNTRUSTED'}",
                                f"fx {K[0,0]:.0f} fy {K[1,1]:.0f} cx {K[0,2]:.0f} cy {K[1,2]:.0f}",
                                f"fov {fovx:.1f} x {fovy:.1f} deg", f"wrote {out}"]
                print("\n".join(result_lines), flush=True)
                print("K =\n", np.round(K, 2)); print("dist =", np.round(dist, 5))
        else:
            vis = img.copy()
            cv2.rectangle(vis, (0, 0), (w, 60 + 45 * len(result_lines)), (0, 200, 0), -1)
            for i, line in enumerate(result_lines):
                cv2.putText(vis, line, (20, 45 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        okj, buf = cv2.imencode(".jpg", cv2.resize(vis, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 70])
        if okj:
            State.jpeg = buf.tobytes()
        if grabbed >= a.frames and time.time() - last > 8:
            break
    cap.release()


if __name__ == "__main__":
    main()

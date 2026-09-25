#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["opencv-python-headless", "numpy"]
# ///
"""Live view of the wrist camera in a browser, with a focus score.

    uv run wrist_cam.py                 # opens http://127.0.0.1:8765 in the browser
    uv run wrist_cam.py --index 1       # a different UVC device
    uv run wrist_cam.py --snap ref.jpg  # save one frame and exit, no server

The board camera on the G1 wrist is a 32 mm UVC module with a wide M12 lens,
mounted upside down, streaming 1920 x 1080. Frames are rotated 180 deg by
default so the image is upright; the recorder has to apply the same rotation.

The green number is the variance of the Laplacian of the grey frame. It is
only meaningful relative to itself: turn the M12 barrel until it peaks and the
picture is crisp. Out of focus the wrist module reads ~3, focused it reads
well above 100 on an ordinary room.

macOS: the terminal needs Camera access (System Settings > Privacy & Security
> Camera) or OpenCV reports "camera failed to properly initialize".
"""
import argparse
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

PAGE = (b'<html><body style="margin:0;background:#000">'
        b'<img src="/mjpg" style="width:100vw;height:100vh;object-fit:contain">'
        b'</body></html>')


def open_camera(index, width, height):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        sys.exit(f"camera {index} did not open (macOS: grant the terminal Camera access)")
    for _ in range(10):  # let exposure settle
        cap.read()
    return cap


def sharpness(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=0, help="OpenCV device index (default 0)")
    ap.add_argument("--width", type=int, default=0, help="request a capture width (0 = driver default)")
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--no-rotate", action="store_true", help="do not rotate the frame 180 deg")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    ap.add_argument("--snap", metavar="PATH", help="save one (rotated) frame to PATH and exit")
    a = ap.parse_args()

    cap = open_camera(a.index, a.width, a.height)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {a.index}: {w}x{h}, rotate {'off' if a.no_rotate else '180'}")

    def read():
        ok, f = cap.read()
        if ok and not a.no_rotate:
            f = cv2.rotate(f, cv2.ROTATE_180)
        return ok, f

    if a.snap:
        ok, f = read()
        if not ok:
            sys.exit("no frame")
        cv2.imwrite(a.snap, f)
        print(f"saved {a.snap}  sharpness {sharpness(f):.1f}")
        return

    state = {"jpg": None, "best": 0.0}
    lock = threading.Lock()

    def grab():
        while True:
            ok, f = read()
            if not ok:
                time.sleep(0.05); continue
            s = sharpness(f); state["best"] = max(state["best"], s)
            cv2.putText(f, f"sharpness {s:6.1f}   best {state['best']:6.1f}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            _, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with lock:
                state["jpg"] = jpg.tobytes()

    threading.Thread(target=grab, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path != "/mjpg":
                self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
                self.wfile.write(PAGE); return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            try:
                while True:
                    with lock:
                        b = state["jpg"]
                    if b:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(b))
                        self.wfile.write(b + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass

    url = f"http://127.0.0.1:{a.port}/"
    print(f"serving {url}  (Ctrl-C to stop)")
    if not a.no_open:
        webbrowser.open(url)
    try:
        HTTPServer(("127.0.0.1", a.port), H).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()

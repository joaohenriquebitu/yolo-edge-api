#!/usr/bin/env python3
import argparse
import subprocess
import threading
import time
from flask import Flask, Response

app = Flask(__name__)
lock = threading.Lock()
_latest_jpg: bytes = b""

def _capture_loop(device: int, width: int, height: int, fps: int):
    global _latest_jpg
    cmd = [
        "rpicam-vid", "-t", "0", "-n", "--codec", "mjpeg",
        "--camera", str(device),
        "--width", str(width), "--height", str(height),
        "--framerate", str(fps),
        "-o", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    buf = b""
    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        end = buf.rfind(b"\xff\xd9")
        if end == -1:
            continue
        start = buf.rfind(b"\xff\xd8", 0, end)
        if start == -1:
            continue
        jpg = buf[start:end + 2]
        buf = buf[end + 2:]
        with lock:
            _latest_jpg = jpg

def _generate_mjpeg():
    while True:
        with lock:
            jpg = _latest_jpg
        if not jpg:
            time.sleep(0.01)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

@app.route('/')
def index():
    return "<html><body><h1>Preview bruto (sem YOLO)</h1><img src='/stream' /></body></html>"

@app.route('/stream')
def stream():
    return Response(_generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot')
def snapshot():
    with lock:
        jpg = _latest_jpg
    if not jpg:
        return Response(status=503)
    return Response(jpg, mimetype='image/jpeg')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--host", type=str, default="0.0.0.0")
    return p.parse_args()

def main():
    args = parse_args()
    t = threading.Thread(
        target=_capture_loop,
        args=(args.device, args.width, args.height, args.fps),
        daemon=True
    )
    t.start()
    time.sleep(1.0)
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
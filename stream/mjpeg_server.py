#!/usr/bin/env python3
"""
stream/mjpeg_server.py
Serve o stream YOLO anotado como MJPEG via HTTP.
"""
import argparse
import threading
import time
import sys
import json
from pathlib import Path
import cv2
from flask import Flask, Response

sys.path.insert(0, str(Path(__file__).parent.parent))
from stream.v3_optimized import OptimizedCamera, RealtimeDetector

app = Flask(__name__)
camera = None
detector = None
lock = threading.Lock()
_latest_jpg: bytes = b""

def _frame_producer():
    global _latest_jpg
    while True:
        frame = camera.read(timeout=2.0)
        if frame is None:
            continue
        annotated = detector.process(frame)
        ok, jpg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with lock:
                _latest_jpg = jpg.tobytes()

def _generate_mjpeg():
    while True:
        with lock:
            jpg = _latest_jpg
        if not jpg:
            time.sleep(0.01)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
        time.sleep(0.033)

@app.route('/')
def index():
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>YOLO Stream em Tempo Real</title>
<style>
body { background: #111; color:#eee; font-family:sans-serif; display: flex; flex-direction:column; align-items:center; padding-top:30px; }
h1 { margin-bottom: 12px; font-size: 1.4rem; }
img { border: 2px solid #444; border-radius: 4px; max-width:100%; }
p { color: #888; font-size: 0.85rem; margin-top: 10px; }
</style>
</head>
<body>
<h1>YOLOv8 — Raspberry Pi 5 Tempo Real</h1>
<img src='/stream' />
<p>Stream MJPEG com inferência YOLO e anotações em tempo real.</p>
</body>
</html>"""

@app.route('/stream')
def stream():
    return Response(_generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    status = {
        "status": "ok",
        "stream": "active",
        "frame_count": detector._frame_idx if detector else 0,
    }
    return Response(json.dumps(status), mimetype='application/json')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--model", type=str, default="models/yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--infer-every", type=int, default=3)
    p.add_argument("--infer-size", type=int, default=320)
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", type=str, default="0.0.0.0")
    return p.parse_args()

def main():
    global camera, detector
    args = parse_args()
    camera = OptimizedCamera(args.device, args.width, args.height, args.fps).start()
    detector = RealtimeDetector(args.model, args.conf, args.infer_every, args.infer_size)

    producer = threading.Thread(target=_frame_producer, daemon=True, name="Frame Producer")
    producer.start()
    time.sleep(1.0)

    print(f"[INFO] Servidor MJPEG iniciado na porta {args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()

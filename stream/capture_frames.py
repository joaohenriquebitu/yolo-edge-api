#!/usr/bin/env python3
import argparse
import time
import urllib.request
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

OUTPUT_DIR = Path("dataset/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_snapshot(url: str):
    try:
        req = urllib.request.urlopen(url, timeout=3)
        data = req.read()
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame is not None, frame
    except Exception as e:
        print(f"[ERRO] Falha ao acessar snapshot: {e}")
        return False, None

def is_sharp_enough(frame: np.ndarray, threshold: float = 20.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score >= threshold

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-url", default="http://localhost:5001/snapshot")
    p.add_argument("--total", type=int, default=150)
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()

    saved = 0
    skipped = 0
    print(f"[INFO] Coletando {args.total} frames de {args.snapshot_url}")

    try:
        while saved < args.total:
            ret, frame = fetch_snapshot(args.snapshot_url)
            
            if not ret or frame is None:
                print("[AVISO] Servidor sem resposta. Aguardando 1s...")
                time.sleep(1.0)
                continue

            if not is_sharp_enough(frame):
                skipped += 1
                print(f"[DESCARTE] Frame borrado (Total descartados: {skipped})")
                time.sleep(args.interval)
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = OUTPUT_DIR / f"frame_{ts}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
            print(f"[{saved}/{args.total}] Salvo: {path.name} (Descartados: {skipped})")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[INFO] Captura interrompida pelo usuario.")

    print(f"\n[OK] {saved} frames salvos em {OUTPUT_DIR} (Descartados: {skipped})")

if __name__ == "__main__":
    main()
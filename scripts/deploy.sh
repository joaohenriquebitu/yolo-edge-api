#!/usr/bin/env bash
set -e

echo "[1/4] Baixando nova imagem..."
docker compose pull
python3 -m dvc pull models/yolo-epi.pt

echo "[2/4] Iniciando nova versao..."
docker compose up -d --build

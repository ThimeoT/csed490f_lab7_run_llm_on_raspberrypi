#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt
# Un seul worker : sessions en mémoire + un seul modèle chargé
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
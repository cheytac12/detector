#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Sign Language App — macOS / Linux launcher
# Handles Python 3.9, 3.10, 3.11, 3.12 automatically
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "┌──────────────────────────────────────────┐"
echo "│   Sign Language Detection App            │"
echo "└──────────────────────────────────────────┘"

# ── 1. Find Python ───────────────────────────────────────────
PYTHON=""
for cmd in python3.11 python3.10 python3.9 python3.12 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "✗ Python 3.9–3.12 not found."
    echo "  Install from https://python.org and rerun."
    exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python $PY_VERSION  →  $PYTHON"

# ── 2. Virtual environment ───────────────────────────────────
VENV=".venv"
if [ ! -d "$VENV" ]; then
    echo "  Creating virtual environment…"
    "$PYTHON" -m venv "$VENV"
fi

PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
source "$VENV/bin/activate"

# ── 3. Install deps with version-aware logic ─────────────────
echo "  Checking / installing dependencies…"

# mediapipe + numpy must match regardless of TF version
"$PIP" install --quiet --upgrade pip

# Python 3.12 needs TF 2.16+ (2.15 wheels don't exist for 3.12)
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -ge 12 ]; then
    echo "  Python 3.12 detected — installing tensorflow 2.16+ …"
    "$PIP" install --quiet \
        "tensorflow>=2.16.0" \
        "mediapipe==0.10.21" \
        "opencv-python==4.9.0.80" \
        "opencv-contrib-python==4.9.0.80" \
        "numpy==1.26.4" \
        "Pillow==10.4.0" \
        "scikit-learn==1.4.2" \
        "scipy==1.13.1"
else
    "$PIP" install --quiet -r requirements.txt
fi

# ── 4. Copy model/data from parent if present ─────────────────
for f in model.h5 model_weights.h5; do
    if [ -f "../$f" ] && [ ! -f "$f" ]; then
        cp "../$f" "$f" && echo "  Copied $f from parent directory."
    fi
done

if [ -d "../MP_Data" ] && [ ! -d "MP_Data" ]; then
    cp -r "../MP_Data" "MP_Data"
    echo "  Copied MP_Data/ from parent directory."
fi

# ── 5. Launch ─────────────────────────────────────────────────
echo ""
echo "  Starting app…"
echo ""
"$PY" app.py

# Sign Language App — Dependency & Setup Guide

## Required Python version

**Python 3.9, 3.10, or 3.11 only.**

`mediapipe` does not support Python 3.12+.  
Check your version: `python --version`

---

## Step 1 — Install system dependencies

### tkinter (must be installed via OS, not pip)

**Windows:**
Reinstall Python from https://python.org.  
During setup, make sure **"tcl/tk and IDLE"** is checked.

**Ubuntu / Debian:**
```bash
sudo apt install python3-tk
```

**Fedora / RHEL:**
```bash
sudo dnf install python3-tkinter
```

**macOS (Homebrew):**
```bash
brew install python-tk@3.11
```

---

## Step 2 — Create a virtual environment

```bash
python -m venv .venv

# Activate — Windows:
.venv\Scripts\activate

# Activate — macOS/Linux:
source .venv/bin/activate
```

---

## Step 3 — Install Python packages

```bash
pip install -r requirements.txt
```

If you get conflicts, install individually:
```bash
pip install numpy==1.26.4
pip install mediapipe==0.10.21
pip install tensorflow==2.15.1
pip install opencv-python==4.9.0.80
pip install Pillow==10.4.0
pip install scikit-learn==1.4.2
pip install scipy==1.13.1
```

---

## Step 4 — Run the app

```bash
python app.py
```

Or on Windows, double-click `run_windows.bat`.

---

## DroidCam setup

DroidCam lets your Android phone act as a webcam.

### USB mode (recommended — lowest latency)
1. Install DroidCam on your phone (Android/iOS).
2. Install the DroidCam PC client: https://www.dev47apps.com/
3. Connect phone via USB and enable USB debugging.
4. Open DroidCam PC client — it creates a virtual camera device.
5. In the app, open **Settings → Camera source** and enter:
   - `1` or `2` (whichever index DroidCam shows up as)
   - Use the **🔍 Scan** button to auto-detect.

### WiFi mode
1. Open DroidCam on your phone and note the IP shown on screen.
2. In the app, open **Settings → Camera source** and enter:
   ```
   http://192.168.x.x:4747/video
   ```
   Replace `192.168.x.x` with your phone's actual IP.
3. Save settings. Use this same URL in the Camera selector on any tab.

### Troubleshooting DroidCam
- If camera index 0 is your laptop webcam and DroidCam is not detected,
  try indices 1, 2, 3 using the **🔍 Scan** button.
- If WiFi mode shows a black screen, make sure phone and PC are on the
  same WiFi network and no firewall is blocking port 4747.
- For lowest latency during data collection, prefer USB mode.

---

## Verifying all dependencies

Run this in your venv to check everything is installed:

```python
import tkinter; print("tkinter OK")
import cv2; print("cv2 OK:", cv2.__version__)
import numpy; print("numpy OK:", numpy.__version__)
import mediapipe; print("mediapipe OK:", mediapipe.__version__)
import tensorflow; print("tensorflow OK:", tensorflow.__version__)
from PIL import Image; print("Pillow OK")
import sklearn; print("scikit-learn OK:", sklearn.__version__)
```

---

## v2/main.py (CLI version) dependencies

The CLI app (`v2/main.py`) requires the same packages.
To run it:
```bash
python v2/main.py
```
All four modes (collect, train, live detect, video detect) work identically
to the GUI — the GUI is a visual wrapper around the same logic.

---

## Common errors and fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'mediapipe'` | `pip install mediapipe==0.10.21` |
| `ModuleNotFoundError: No module named 'tkinter'` | Install via OS (see Step 1) |
| `ModuleNotFoundError: No module named 'cv2'` | `pip install opencv-python==4.9.0.80` |
| `ModuleNotFoundError: No module named 'PIL'` | `pip install Pillow` |
| `Cannot open camera: 0` | Try index 1 or 2, or use DroidCam URL |
| mediapipe crashes with numpy 2.x | `pip install numpy==1.26.4` |
| TensorFlow not found | `pip install tensorflow==2.15.1` (Python ≤ 3.11 required) |

# Sign Language Detection — Desktop App

All original notebook + v2/main.py functions in a clean dark GUI.

## Quick start

**macOS / Linux**
```bash
chmod +x run_mac_linux.sh && ./run_mac_linux.sh
```

**Windows** — double-click `run_windows.bat`

The launchers create a venv, detect your Python version, install the right dependencies, and start the app. No manual pip installs needed.

---

## What's preserved from the originals

| Original | Preserved |
|---|---|
| `mediapipe_detection()` | ✓ Exact copy |
| `draw_styled_landmarks()` | ✓ Exact copy — all 4 colours |
| `extract_keypoints()` → 1662-dim | ✓ Exact copy |
| `prob_viz()` overlay on frame | ✓ Exact copy — orange bars on frame |
| LSTM 64→128→64→Dense architecture | ✓ Exact copy |
| TensorBoard callback during training | ✓ Preserved (run `tensorboard --logdir=Logs`) |
| `multilabel_confusion_matrix` + `accuracy_score` | ✓ Evaluation tab |
| Sentence logic: `predictions[-10:]` stability check | ✓ Exact copy in `SentenceBuilder` |
| Sentence orange banner overlay on video frame | ✓ Exact copy |
| `model.save()` + `model.save_weights()` | ✓ Preserved |
| Video file detection | ✓ Video Detect tab |
| GPU toggle | ✓ Settings / Train tab |

---

## App tabs

| Tab | What it does |
|---|---|
| **Camera Preview** | Raw webcam with styled landmarks — no inference |
| **Collect Data** | Record landmark sequences; shows live preview |
| **Train Model** | LSTM training with live log + TensorBoard support |
| **Live Detect** | Real-time inference, prob bars, sentence builder |
| **Video Detect** | Same inference on a video file |
| **Evaluation** | Confusion matrix + accuracy on held-out 5% split |
| **Settings** | Threshold, paths, GPU toggle — persisted in `app_settings.json` |

---

## Python / dependency compatibility

| Python | TensorFlow installed |
|---|---|
| 3.9, 3.10, 3.11 | 2.15.1 (stable, tested) |
| 3.12 | 2.16+ (auto-detected by launcher) |

The requirements.txt pins:
- `numpy==1.26.4` — mediapipe 0.10.x breaks with numpy 2.x
- `opencv-python==4.9.0.80` — stable, no conflicts
- `mediapipe==0.10.21` — latest 0.10 series
- `scikit-learn==1.4.2` — stable sklearn
- `Pillow==10.4.0` — for tkinter image display

---

## Using your existing model

Place `model.h5`, `model_weights.h5`, and `MP_Data/` in the same folder as `app.py`.  
The app auto-detects classes from `MP_Data/` and loads the model on startup.

---

## TensorBoard

After training starts, open a second terminal and run:
```bash
tensorboard --logdir=Logs
```
Then open `http://localhost:6006` in a browser.

---

## Folder structure

```
sign_language_app/
├── app.py
├── requirements.txt
├── run_mac_linux.sh
├── run_windows.bat
├── model.h5
├── model_weights.h5
├── app_settings.json   ← created on first save
├── Logs/               ← TensorBoard logs
└── MP_Data/
    ├── hello/
    ├── thanks/
    └── iloveyou/
```

#!/usr/bin/env python3
"""
Sign Language Detection - Desktop Application
Preserves ALL original notebook + v2/main.py functionality:
  * Webcam landmark preview (draw_styled_landmarks)
  * Dataset collection (with per-sign countdown)
  * Model training (LSTM 64->128->64->Dense, TensorBoard, save/load)
  * Evaluation (multilabel_confusion_matrix, accuracy_score)
  * Live detection with sentence builder + prob_viz overlay
  * Video file detection
  * GPU toggle
  * Auto class detection from MP_Data/
  * Camera source selector (index or IP/URL - supports DroidCam)
"""

import sys
if sys.version_info < (3, 8):
    print(
        "\n[ERROR] Python {}.{} detected. Python 3.8 or newer is required.\n"
        "  mediapipe 0.10.x does not support Python 3.7.\n"
        "  Please install Python 3.9, 3.10, or 3.11 from https://python.org\n"
        "  then re-run: python app.py\n".format(*sys.version_info[:2])
    )
    input("Press Enter to exit...")
    sys.exit(1)

import os
import sys
import threading
import time
import queue
import json
import platform
from collections import deque
from pathlib import Path

# suppress TF noise before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# -- tkinter: give a clear error if missing ----------------------------------
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog, scrolledtext
except ModuleNotFoundError:
    print(
        "\n[ERROR] tkinter is not installed.\n"
        "  On Debian/Ubuntu:  sudo apt install python3-tk\n"
        "  On Fedora/RHEL:    sudo dnf install python3-tkinter\n"
        "  On Windows/macOS:  reinstall Python from python.org (enable tk/tcl option)\n"
    )
    sys.exit(1)

# -- opencv -------------------------------------------------------------------
try:
    import cv2
except ModuleNotFoundError:
    print(
        "\n[ERROR] opencv-python is not installed.\n"
        "  Run: pip install opencv-python\n"
    )
    sys.exit(1)

# -- numpy --------------------------------------------------------------------
try:
    import numpy as np
except ModuleNotFoundError:
    print("\n[ERROR] numpy not installed. Run: pip install numpy==1.26.4\n")
    sys.exit(1)

# -- Pillow --------------------------------------------------------------------
try:
    from PIL import Image, ImageTk
except ModuleNotFoundError:
    print(
        "\n[ERROR] Pillow is not installed.\n"
        "  Run: pip install Pillow\n"
    )
    sys.exit(1)

# -----------------------------------------------------------------
# CONSTANTS  (match originals exactly)
# -----------------------------------------------------------------
DATA_PATH       = "MP_Data"
MODEL_PATH      = "model.h5"
WEIGHTS_PATH    = "model_weights.h5"
SETTINGS_FILE   = "app_settings.json"
SEQ_LEN         = 30
SEQS_PER_SIGN   = 30
FEAT_VEC        = 1662
DEFAULT_THRESH  = 0.5
ERROR_MSG_MAX_LEN = 120
DEFAULT_EPOCHS  = 2000
LOG_DIR         = "Logs"

# -----------------------------------------------------------------
# COLOUR PALETTE
# -----------------------------------------------------------------
C = dict(
    bg        = "#0d0d0d",
    surface   = "#161616",
    surface2  = "#1e1e1e",
    border    = "#2c2c2c",
    accent    = "#c9a96e",
    accent_dk = "#7a6340",
    text      = "#e6e6e6",
    muted     = "#5a5a5a",
    success   = "#4e9e6e",
    danger    = "#b85555",
    info      = "#5580b8",
    warn      = "#b89055",
    face_c    = (80, 110, 10),
    pose_c    = (80, 22, 10),
    lh_c      = (121, 22, 76),
    rh_c      = (245, 117, 66),
)

IS_MAC = platform.system() == "Darwin"
FONT   = "SF Pro Text" if IS_MAC else "Segoe UI"
MONO   = "SF Mono"     if IS_MAC else "Consolas"


def F(size, bold=False):
    return (FONT, size, "bold" if bold else "normal")


def FM(size):
    return (MONO, size, "normal")


# -----------------------------------------------------------------
# CAMERA SOURCE HELPERS
# -----------------------------------------------------------------
def parse_camera_source(src: str):
    """
    Convert a user-supplied camera string to the right type for cv2.VideoCapture.
    Accepts:
      - integer index:  "0", "1", "2" ...
      - DroidCam URL:   "http://192.168.x.x:4747/video"
      - Any RTSP/HTTP URL string
    """
    src = src.strip()
    try:
        return int(src)
    except ValueError:
        return src   # string URL - cv2 accepts these directly


def probe_cameras(max_index: int = 6) -> list:
    """
    Return a list of camera index strings that cv2 can open.
    Probes indices 0..max_index-1 quickly without blocking.
    """
    found = []
    for i in range(max_index):
        cap = open_video_capture(i)
        if cap is not None and cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(str(i))
            cap.release()
    return found


def open_video_capture(src):
    """
    Open a camera/video source with platform-aware fallbacks.
    On Windows camera indices often work better with DirectShow.
    URL sources keep OpenCV defaults.
    """
    if isinstance(src, int) and platform.system() == "Windows":
        tried = []
        for name in ("CAP_DSHOW", "CAP_MSMF"):
            backend = getattr(cv2, name, None)
            if backend is None:
                continue
            cap = cv2.VideoCapture(src, backend)
            tried.append(cap)
            if cap is not None and cap.isOpened():
                return cap
        for cap in tried:
            if cap is not None:
                cap.release()
    return cv2.VideoCapture(src)


# -----------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------
def load_settings():
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


# -----------------------------------------------------------------
# AUTO CLASS DETECTION
# -----------------------------------------------------------------
def detect_classes(data_path=DATA_PATH):
    p = Path(data_path)
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir()
                  if d.is_dir() and any(d.iterdir()))


def count_sequences(sign, data_path=DATA_PATH):
    p = Path(data_path, sign)
    if not p.exists():
        return 0
    return len([d for d in p.iterdir() if d.is_dir()])


# -----------------------------------------------------------------
# MEDIAPIPE HELPERS  (exact originals)
# -----------------------------------------------------------------
class MP:
    """
    Wraps mediapipe imports so they load lazily and only once.

    mediapipe 0.10.x on some builds exposes solutions as a direct
    subpackage rather than an attribute of the top-level module.
    We import mediapipe.solutions.holistic and
    mediapipe.solutions.drawing_utils directly to avoid the
    'module has no attribute solutions' error.
    """
    _holistic_mod  = None
    _drawing_mod   = None

    @classmethod
    def holistic_mod(cls):
        if cls._holistic_mod is None:
            try:
                # Direct submodule import — works across all 0.10.x builds
                import mediapipe.solutions.holistic as _h
                cls._holistic_mod = _h
            except ImportError:
                try:
                    import mediapipe as mp
                    cls._holistic_mod = mp.solutions.holistic
                except Exception:
                    raise ModuleNotFoundError(
                        "mediapipe is not installed or is broken.\n"
                        "  Run: pip install mediapipe==0.10.21 numpy==1.26.4\n"
                        "  NOTE: mediapipe requires Python 3.8-3.11 and numpy < 2.0"
                    )
        return cls._holistic_mod

    @classmethod
    def drawing(cls):
        if cls._drawing_mod is None:
            try:
                import mediapipe.solutions.drawing_utils as _d
                cls._drawing_mod = _d
            except ImportError:
                try:
                    import mediapipe as mp
                    cls._drawing_mod = mp.solutions.drawing_utils
                except Exception:
                    raise ModuleNotFoundError(
                        "mediapipe drawing_utils not found.\n"
                        "  Run: pip install mediapipe==0.10.21"
                    )
        return cls._drawing_mod

    @classmethod
    def new_holistic(cls, det=0.5, track=0.5):
        return cls.holistic_mod().Holistic(
            min_detection_confidence=det,
            min_tracking_confidence=track,
        )


def mediapipe_detection(image, model):
    """Exact replica of original notebook function."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def _get_face_connections():
    """
    mediapipe 0.10.x moved FACEMESH_TESSELATION out of the Holistic module
    in some builds. This helper finds the right attribute across versions.
    Falls back to FACEMESH_CONTOURS or None (which skips connection lines
    but still draws the landmark dots).
    """
    mp_holistic = MP.holistic_mod()
    if hasattr(mp_holistic, "FACEMESH_TESSELATION"):
        return mp_holistic.FACEMESH_TESSELATION
    # Try direct face_mesh submodule import (mediapipe 0.10.x)
    try:
        import mediapipe.solutions.face_mesh as fm
        if hasattr(fm, "FACEMESH_TESSELATION"):
            return fm.FACEMESH_TESSELATION
        if hasattr(fm, "FACEMESH_CONTOURS"):
            return fm.FACEMESH_CONTOURS
    except ImportError:
        pass
    if hasattr(mp_holistic, "FACEMESH_CONTOURS"):
        return mp_holistic.FACEMESH_CONTOURS
    return None


def draw_styled_landmarks(image, results):
    """Exact replica of original notebook draw_styled_landmarks."""
    mp_drawing = MP.drawing()
    mp_holistic = MP.holistic_mod()
    Spec = mp_drawing.DrawingSpec

    face_connections = _get_face_connections()
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, face_connections,
        Spec(color=(80, 110, 10), thickness=1, circle_radius=1),
        Spec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        Spec(color=(80, 22, 10), thickness=2, circle_radius=4),
        Spec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        Spec(color=(121, 22, 76), thickness=2, circle_radius=4),
        Spec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        Spec(color=(245, 117, 66), thickness=2, circle_radius=4),
        Spec(color=(245, 66, 230), thickness=2, circle_radius=2))


def extract_keypoints(results):
    """Exact replica of original extract_keypoints -> 1662-dim vector."""
    pose = (np.array([[r.x, r.y, r.z, r.visibility]
                      for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4))
    face = (np.array([[r.x, r.y, r.z]
                      for r in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks else np.zeros(468 * 3))
    lh   = (np.array([[r.x, r.y, r.z]
                      for r in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(21 * 3))
    rh   = (np.array([[r.x, r.y, r.z]
                      for r in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(21 * 3))
    return np.concatenate([pose, face, lh, rh])


def prob_viz(res, actions, input_frame, colors):
    """Exact replica of original prob_viz overlay."""
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        color = colors[num % len(colors)]
        cv2.rectangle(output_frame,
                      (0, 60 + num * 40), (int(prob * 100), 90 + num * 40),
                      color, -1)
        cv2.putText(output_frame, actions[num],
                    (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame


def build_lstm_model(num_classes, seq_len=SEQ_LEN, feat=FEAT_VEC):
    """Exact LSTM architecture from original notebooks."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    model = Sequential([
        LSTM(64,  return_sequences=True,  activation='relu',
             input_shape=(seq_len, feat)),
        LSTM(128, return_sequences=True,  activation='relu'),
        LSTM(64,  return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


# -----------------------------------------------------------------
# SENTENCE BUILDER  (from original notebook logic)
# -----------------------------------------------------------------
class SentenceBuilder:
    def __init__(self, threshold=DEFAULT_THRESH, max_len=5, min_streak=6):
        self.sentence    : list = []
        self.predictions : list = []
        self.last_word           = None
        self.threshold           = threshold
        self.max_len             = max_len
        self.min_streak          = max(1, int(min_streak))
        self._streak_idx         = None
        self._streak_count       = 0

    def update(self, res, actions):
        idx = int(np.argmax(res))
        self.predictions.append(idx)

        if self._streak_idx == idx:
            self._streak_count += 1
        else:
            self._streak_idx = idx
            self._streak_count = 1

        added = False
        if self._streak_count >= self.min_streak and res[idx] >= self.threshold:
            word = actions[idx]
            if word != self.last_word:
                self.last_word = word
                if word not in ("neutral", "idle"):
                    self.sentence.append(word)
                    added = True

        if len(self.sentence) > self.max_len:
            self.sentence = self.sentence[-self.max_len:]
        if len(self.predictions) > 200:
            self.predictions = self.predictions[-200:]

        return added

    def text(self):
        return " ".join(self.sentence)

    def undo(self):
        if self.sentence:
            self.sentence.pop()
            self.last_word = self.sentence[-1] if self.sentence else None

    def clear(self):
        self.sentence    = []
        self.predictions = []
        self.last_word   = None
        self._streak_idx = None
        self._streak_count = 0


# -----------------------------------------------------------------
# BACKGROUND THREADS
# -----------------------------------------------------------------

class CameraPreviewThread(threading.Thread):
    """Tab 1 - live webcam with styled landmarks, no inference."""
    def __init__(self, out_q, cam_src=0):
        super().__init__(daemon=True)
        self.out_q   = out_q
        self.cam_src = cam_src
        self.running = True
        self.error   = None

    def run(self):
        try:
            cap = open_video_capture(self.cam_src)
            if not cap.isOpened():
                self.error = f"Cannot open camera: {self.cam_src!r}"
                return
            with MP.new_holistic() as hm:
                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.03)
                        continue
                    image, results = mediapipe_detection(frame, hm)
                    draw_styled_landmarks(image, results)
                    try:
                        self.out_q.put_nowait(image)
                    except queue.Full:
                        pass
            cap.release()
        except Exception as e:
            self.error = str(e)

    def stop(self):
        self.running = False


class CollectThread(threading.Thread):
    """Tab 2 - dataset collection, exact original logic."""
    def __init__(self, signs, seq_len, n_seqs, status_q, frame_q, cam_src=0):
        super().__init__(daemon=True)
        self.signs    = signs
        self.seq_len  = seq_len
        self.n_seqs   = n_seqs
        self.status_q = status_q
        self.frame_q  = frame_q
        self.cam_src  = cam_src
        self.running  = True
        self.error    = None

    def run(self):
        try:
            Path(DATA_PATH).mkdir(exist_ok=True)
            for sign in self.signs:
                for seq in range(self.n_seqs):
                    Path(DATA_PATH, sign, str(seq)).mkdir(parents=True, exist_ok=True)

            total_seqs = len(self.signs) * self.n_seqs
            done_seqs  = 0

            cap = open_video_capture(self.cam_src)
            if not cap.isOpened():
                self.status_q.put(dict(error=f"Cannot open camera: {self.cam_src!r}"))
                return

            with MP.new_holistic() as hm:
                for sign in self.signs:
                    if not self.running:
                        break
                    for seq in range(self.n_seqs):
                        if not self.running:
                            break
                        for frame_num in range(self.seq_len):
                            if not self.running:
                                break

                            ret, frame = cap.read()
                            if not ret:
                                continue

                            image, results = mediapipe_detection(frame, hm)
                            draw_styled_landmarks(image, results)

                            if frame_num == 0:
                                cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                                cv2.putText(image,
                                            f'Collecting: {sign}  seq {seq}', (15, 12),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                try:
                                    self.frame_q.put_nowait(image)
                                except queue.Full:
                                    pass
                                time.sleep(0.5)
                            else:
                                cv2.putText(image,
                                            f'Collecting: {sign}  seq {seq}  frame {frame_num}', (15, 12),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                            kp = extract_keypoints(results)
                            np.save(os.path.join(DATA_PATH, sign, str(seq), f"{frame_num}.npy"), kp)

                            try:
                                self.frame_q.put_nowait(image)
                            except queue.Full:
                                pass

                            pct = int(done_seqs / total_seqs * 100)
                            self.status_q.put(dict(
                                sign=sign, seq=seq + 1, frame=frame_num + 1,
                                total_seqs=total_seqs, done_seqs=done_seqs,
                                pct=pct))

                        done_seqs += 1

            cap.release()
            self.status_q.put(dict(done_all=True))
        except Exception as e:
            import traceback
            self.status_q.put(dict(error=traceback.format_exc()))

    def stop(self):
        self.running = False


class TrainThread(threading.Thread):
    """Tab 3 - training, exact original logic + TensorBoard callback."""
    def __init__(self, signs, epochs, seq_len, log_q, use_gpu=True):
        super().__init__(daemon=True)
        self.signs   = signs
        self.epochs  = epochs
        self.seq_len = seq_len
        self.log_q   = log_q
        self.use_gpu = use_gpu
        self.running = True

    def run(self):
        try:
            if not self.use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

            import tensorflow as tf
            from tensorflow.keras.utils import to_categorical
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

            self._log("Loading dataset...")
            label_map  = {s: i for i, s in enumerate(self.signs)}
            sequences, labels = [], []

            for sign in self.signs:
                sign_dir = Path(DATA_PATH, sign)
                try:
                    seq_ids = sorted(
                        int(x) for x in os.listdir(sign_dir)
                        if Path(sign_dir, x).is_dir())
                except Exception:
                    continue
                for seq in seq_ids:
                    window = []
                    for f in range(self.seq_len):
                        fp = Path(DATA_PATH, sign, str(seq), f"{f}.npy")
                        if fp.exists():
                            window.append(np.load(fp))
                    if len(window) == self.seq_len:
                        sequences.append(window)
                        labels.append(label_map[sign])

            if not sequences:
                self._log("FAIL No complete sequences found. Collect data first.")
                self.log_q.put(("error", "No data"))
                return

            X = np.array(sequences)
            y = to_categorical(labels).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
            self._log(f"Dataset: {len(X)} sequences | {len(self.signs)} classes")
            self._log(f"Train: {len(X_train)}  Test: {len(X_test)}")

            Path(LOG_DIR).mkdir(exist_ok=True)
            tb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

            model = build_lstm_model(len(self.signs), self.seq_len)
            model.summary(print_fn=lambda s: self._log(s))

            self._log(f"\nTraining for {self.epochs} epochs...")
            self._log("(TensorBoard: run  tensorboard --logdir=Logs)\n")

            class ProgressCB(tf.keras.callbacks.Callback):
                def __init__(cb, q, total):
                    super().__init__()
                    cb.q = q
                    cb.total = total

                def on_epoch_end(cb, epoch, logs=None):
                    if not self.running:
                        cb.model.stop_training = True
                        return
                    ep1 = epoch + 1
                    if ep1 % 100 == 0 or ep1 <= 5:
                        acc  = logs.get('categorical_accuracy', 0)
                        loss = logs.get('loss', 0)
                        pct  = int(ep1 / cb.total * 100)
                        msg  = f"Ep {ep1}/{cb.total}  loss={loss:.4f}  acc={acc:.4f}"
                        cb.q.put(("progress", pct, msg))

            model.fit(X_train, y_train, epochs=self.epochs,
                      callbacks=[tb, ProgressCB(self.log_q, self.epochs)],
                      verbose=0)

            if not self.running:
                self._log("Training cancelled.")
                self.log_q.put(("cancelled",))
                return

            model.save(MODEL_PATH)
            model.save_weights(WEIGHTS_PATH)
            self._log(f"\nOK Saved {MODEL_PATH} + {WEIGHTS_PATH}")

            yhat_raw  = model.predict(X_test, verbose=0)
            ytrue     = np.argmax(y_test, axis=1).tolist()
            yhat      = np.argmax(yhat_raw, axis=1).tolist()
            acc       = accuracy_score(ytrue, yhat)
            try:
                cm = multilabel_confusion_matrix(ytrue, yhat)
                self._log(f"\nConfusion matrices (one per class):\n{cm}")
            except Exception:
                pass
            self._log(f"\nTest accuracy: {acc:.4f} ({acc:.1%})")
            self.log_q.put(("done", acc))

        except Exception as e:
            import traceback
            self.log_q.put(("error", traceback.format_exc()))

    def _log(self, msg):
        self.log_q.put(("log", msg))

    def stop(self):
        self.running = False


class DetectThread(threading.Thread):
    """Tab 4/5 - real-time inference, exact notebook sentence logic."""
    def __init__(self, model, signs, src, out_q, threshold=DEFAULT_THRESH,
                 show_prob_viz=True):
        super().__init__(daemon=True)
        self.model         = model
        self.signs         = signs
        self.src           = src
        self.out_q         = out_q
        self.threshold     = threshold
        self.show_prob_viz = show_prob_viz
        self.running       = True
        self.colors        = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
        self._seq_buf      = []
        self._prob_hist    = deque(maxlen=5)

    def run(self):
        builder = SentenceBuilder(self.threshold)
        cap = open_video_capture(self.src)

        if not cap.isOpened():
            self.out_q.put({"done": True, "error": f"Cannot open source: {self.src!r}"})
            return

        with MP.new_holistic() as hm:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image, results = mediapipe_detection(frame, hm)
                draw_styled_landmarks(image, results)

                kp = extract_keypoints(results)
                self._seq_buf.append(kp)
                self._seq_buf = self._seq_buf[-SEQ_LEN:]

                pred_word  = None
                confidence = 0.0
                probs      = []

                if len(self._seq_buf) == SEQ_LEN:
                    raw_res = self.model.predict(
                        np.expand_dims(self._seq_buf, axis=0), verbose=0)[0]
                    self._prob_hist.append(raw_res)
                    res = np.mean(np.asarray(self._prob_hist), axis=0)
                    probs      = res.tolist()
                    idx        = int(np.argmax(res))
                    confidence = float(res[idx])
                    pred_word  = self.signs[idx] if confidence >= self.threshold else None

                    builder.update(res, self.signs)

                    if self.show_prob_viz:
                        image = prob_viz(res, self.signs, image, self.colors)

                # orange bar + sentence overlay (exact original)
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, builder.text(), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

                try:
                    self.out_q.put_nowait({
                        "frame":      image,
                        "sentence":   builder.text(),
                        "word":       pred_word,
                        "confidence": confidence,
                        "probs":      probs,
                        "signs":      self.signs,
                        "builder":    builder,
                    })
                except queue.Full:
                    pass

        cap.release()
        self.out_q.put({"done": True})

    def stop(self):
        self.running = False
        self._seq_buf = []
        self._prob_hist.clear()


# -----------------------------------------------------------------
# UI HELPERS
# -----------------------------------------------------------------
def _lighten(hex_c, amt=28):
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(
        min(255, r + amt), min(255, g + amt), min(255, b + amt))


class Btn(tk.Label):
    """Flat clickable button."""
    def __init__(self, parent, text, cmd, bg=None, fg=None, small=False, **kw):
        _bg = bg or C["border"]
        _fg = fg or C["text"]
        f   = F(9) if small else F(10)
        px  = 10 if small else 18
        py  = 4  if small else 7
        super().__init__(parent, text=text, font=f, bg=_bg, fg=_fg,
                         padx=px, pady=py, cursor="hand2",
                         relief="flat", **kw)
        self._bg = _bg
        self.bind("<Button-1>", lambda e: cmd())
        self.bind("<Enter>",    lambda e: self.config(bg=_lighten(_bg)))
        self.bind("<Leave>",    lambda e: self.config(bg=_bg))

    def set_state(self, enabled: bool):
        self.config(fg=C["text"] if enabled else C["muted"],
                    cursor="hand2" if enabled else "arrow")


class SectionLabel(tk.Label):
    def __init__(self, parent, text, **kw):
        super().__init__(parent, text=text, font=F(8), bg=C["bg"],
                         fg=C["muted"], anchor="w", **kw)


def entry_field(parent, var, wide=False):
    w = 0 if wide else 8
    return tk.Entry(parent, textvariable=var, font=F(10),
                    bg=C["surface2"], fg=C["text"],
                    insertbackground=C["text"], bd=0, relief="flat",
                    width=w,
                    highlightthickness=1,
                    highlightbackground=C["border"],
                    highlightcolor=C["accent"])


def show_frame_on_canvas(canvas, frame, _photo_ref):
    """Fit frame into canvas, return new photo reference."""
    canvas.update_idletasks()
    cw = canvas.winfo_width()
    ch = canvas.winfo_height()
    if cw < 4 or ch < 4:
        return _photo_ref
    h, w = frame.shape[:2]
    scale = min(cw / w, ch / h)
    nw, nh = int(w * scale), int(h * scale)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img   = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    canvas.delete("all")
    canvas.create_image(cw // 2, ch // 2, anchor="center", image=photo)
    return photo


# -----------------------------------------------------------------
# CAMERA SELECTOR WIDGET
# -----------------------------------------------------------------
class CameraSelector(tk.Frame):
    """
    A compact row with a combobox + custom entry for camera source.
    Supports numeric indices (0, 1, 2...) and URL strings
    (DroidCam: http://192.168.x.x:4747/video, RTSP, etc.)
    """
    def __init__(self, parent, label="Camera source", initial="0", **kw):
        super().__init__(parent, bg=C["bg"], **kw)
        tk.Label(self, text=label, font=F(9),
                 bg=C["bg"], fg=C["muted"]).pack(side="left")

        self._var = tk.StringVar(value=initial)

        # dropdown for detected indices
        self._combo = ttk.Combobox(
            self, textvariable=self._var,
            font=F(9), width=28, state="normal")
        self._combo.pack(side="left", padx=(6, 4))
        self._combo.bind("<FocusIn>", lambda e: None)   # allow free typing

        Btn(self, "Scan Scan", self._scan, small=True).pack(side="left", padx=(0, 4))

        self._status = tk.Label(self, text="", font=F(8),
                                bg=C["bg"], fg=C["muted"])
        self._status.pack(side="left")

        self._populate_defaults()

    def _populate_defaults(self):
        detected = probe_cameras()
        opts = detected + ["http://192.168.x.x:4747/video (DroidCam)"]
        self._combo["values"] = opts
        current = self._var.get().strip()
        if detected:
            if not current:
                self._var.set(detected[0])
            self._status.config(text=f"({len(detected)} cam(s) found)")
        else:
            self._status.config(text="(no cams detected - enter manually)")

    def _scan(self):
        self._status.config(text="Scanning...", fg=C["warn"])
        self.update_idletasks()

        def _do():
            found = probe_cameras(8)
            opts  = found + ["http://192.168.x.x:4747/video (DroidCam)"]
            self._combo["values"] = opts
            msg = f"({len(found)} cam(s) found)" if found else "(none found)"
            self._status.config(text=msg, fg=C["muted"])

        threading.Thread(target=_do, daemon=True).start()

    def get_source(self):
        """Return parsed source (int or str) ready for cv2.VideoCapture."""
        raw = self._var.get().strip()
        # strip the helper text appended to DroidCam option
        if " (" in raw:
            raw = raw.split(" (")[0]
        return parse_camera_source(raw)


# -----------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sign Language")
        self.geometry("1280x800")
        self.minsize(960, 640)
        self.configure(bg=C["bg"])

        self.settings   = load_settings()
        self.model      = None
        self.signs      = []
        self._photos    = {}

        self._preview_th : CameraPreviewThread = None
        self._collect_th : CollectThread       = None
        self._train_th   : TrainThread         = None
        self._detect_th  : DetectThread        = None
        self._model_loading                 = False
        self._pending_detect_start          = False

        self._preview_q  = queue.Queue(maxsize=2)
        self._collect_fq = queue.Queue(maxsize=2)
        self._collect_sq = queue.Queue()
        self._train_q    = queue.Queue()
        self._detect_q   = queue.Queue(maxsize=2)

        self._build_ui()
        self._auto_load_model()
        self.after(40, self._tick)

    # ==========================================================
    # UI SKELETON
    # ==========================================================
    def _build_ui(self):
        self._style_ttk()

        # -- sidebar -------------------------------------------
        sb = tk.Frame(self, bg=C["surface"], width=220)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        logo_row = tk.Frame(sb, bg=C["surface"])
        logo_row.pack(fill="x", padx=18, pady=(22, 4))
        tk.Label(logo_row, text="[*]", font=F(26, True), bg=C["surface"],
                 fg=C["accent"]).pack(side="left")
        tk.Label(logo_row, text=" SignLang", font=F(20, True),
                 bg=C["surface"], fg=C["text"]).pack(side="left")

        tk.Frame(sb, bg=C["border"], height=1).pack(fill="x", padx=18, pady=10)

        self._nav_btns = {}
        for key, label in [
            ("preview",  "[o]  Camera Preview"),
            ("collect",  "[+]  Collect Data"),
            ("train",    "[~]  Train Model"),
            ("detect",   "[>]  Live Detect"),
            ("video",    "[>]  Video Detect"),
            ("eval",     "[*]  Evaluation"),
            ("settings", "[S]  Settings"),
        ]:
            b = tk.Label(sb, text=label, font=F(10), bg=C["surface"],
                         fg=C["muted"], padx=18, pady=9, anchor="w",
                         cursor="hand2")
            b.pack(fill="x")
            b.bind("<Button-1>", lambda e, k=key: self._show(k))
            b.bind("<Enter>",    lambda e, b=b: b.config(fg=C["text"])
                   if b != self._nav_btns.get(self._active, "") else None)
            b.bind("<Leave>",    lambda e, b=b: b.config(fg=C["muted"])
                   if b != self._nav_btns.get(self._active, "") else None)
            self._nav_btns[key] = b

        self._sb_bottom = tk.Frame(sb, bg=C["surface"])
        self._sb_bottom.pack(side="bottom", fill="x", padx=18, pady=14)
        self._lbl_mstatus = tk.Label(self._sb_bottom, text="No model",
                                     font=F(8), bg=C["surface"], fg=C["muted"], anchor="w")
        self._lbl_mstatus.pack(fill="x")
        self._lbl_classes = tk.Label(self._sb_bottom, text="",
                                     font=F(8), bg=C["surface"], fg=C["muted"], anchor="w",
                                     wraplength=190, justify="left")
        self._lbl_classes.pack(fill="x", pady=(2, 0))

        # -- main area -----------------------------------------
        self._main = tk.Frame(self, bg=C["bg"])
        self._main.pack(side="right", fill="both", expand=True)

        self._pages = {}
        self._active = ""
        self._build_preview_page()
        self._build_collect_page()
        self._build_train_page()
        self._build_detect_page()
        self._build_video_page()
        self._build_eval_page()
        self._build_settings_page()

        self._show("preview")

    def _style_ttk(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TProgressbar", troughcolor=C["border"],
                    background=C["accent"], thickness=5, relief="flat")
        s.configure("TNotebook", background=C["bg"], borderwidth=0)
        # style the combobox
        s.configure("TCombobox",
                    fieldbackground=C["surface2"],
                    background=C["surface2"],
                    foreground=C["text"],
                    selectbackground=C["accent_dk"],
                    selectforeground=C["text"])

    def _show(self, key):
        for k, f in self._pages.items():
            f.pack_forget()
        self._pages[key].pack(fill="both", expand=True)
        for k, b in self._nav_btns.items():
            if k == key:
                b.config(fg=C["accent"], bg=C["bg"])
            else:
                b.config(fg=C["muted"], bg=C["surface"])
        prev = self._active
        self._active = key
        if prev == "preview" and key != "preview":
            self._stop_preview()
        if key == "preview":
            self._start_preview()
        if key == "collect":
            self._refresh_class_list()
        if key == "train":
            self._refresh_train_classes()
        if key == "eval":
            self._refresh_eval_classes()

    # ==========================================================
    # PAGE 1 - CAMERA PREVIEW
    # ==========================================================
    def _build_preview_page(self):
        p = tk.Frame(self._main, bg=C["bg"])
        self._pages["preview"] = p

        self._page_header(p, "Camera Preview",
                          "Live webcam with styled MediaPipe landmarks (no inference)")

        # Camera selector row
        cam_row = tk.Frame(p, bg=C["bg"])
        cam_row.pack(fill="x", padx=24, pady=(0, 8))
        self._preview_cam = CameraSelector(
            cam_row,
            initial=str(self.settings.get("camera_source", "0")))
        self._preview_cam.pack(side="left")

        body = tk.Frame(p, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        self._preview_canvas = tk.Canvas(body, bg="#000", highlightthickness=0)
        self._preview_canvas.pack(fill="both", expand=True)

        ctrl = tk.Frame(p, bg=C["bg"])
        ctrl.pack(fill="x", padx=24, pady=(0, 14))
        Btn(ctrl, "Start Camera", self._start_preview,
            bg=C["success"]).pack(side="left", padx=(0, 8))
        Btn(ctrl, "Stop Camera", self._stop_preview,
            bg=C["danger"]).pack(side="left")

        self._preview_err_lbl = tk.Label(ctrl, text="", font=F(9),
                                         bg=C["bg"], fg=C["danger"])
        self._preview_err_lbl.pack(side="left", padx=(14, 0))

    def _start_preview(self):
        if self._preview_th and self._preview_th.is_alive():
            return
        self._preview_err_lbl.config(text="")
        src = self._preview_cam.get_source()
        self._preview_th = CameraPreviewThread(self._preview_q, cam_src=src)
        self._preview_th.start()

    def _stop_preview(self):
        if self._preview_th:
            self._preview_th.stop()
            self._preview_th = None

    # ==========================================================
    # PAGE 2 - COLLECT DATA
    # ==========================================================
    def _build_collect_page(self):
        p = tk.Frame(self._main, bg=C["bg"])
        self._pages["collect"] = p

        self._page_header(p, "Collect Training Data",
                          "Record MediaPipe landmark sequences for each sign")

        body = tk.Frame(p, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        form = tk.Frame(body, bg=C["bg"], width=300)
        form.pack(side="left", fill="y", padx=(0, 20))
        form.pack_propagate(False)

        # Camera selector
        SectionLabel(form, "CAMERA SOURCE").pack(anchor="w")
        self._collect_cam = CameraSelector(
            form,
            initial=str(self.settings.get("camera_source", "0")))
        self._collect_cam.pack(anchor="w", pady=(3, 12))

        SectionLabel(form, "SIGNS (comma-separated)").pack(anchor="w")
        self._cv_signs = tk.StringVar(
            value=", ".join(detect_classes()) or "hello, thanks, iloveyou, neutral")
        entry_field(form, self._cv_signs, wide=True).pack(
            fill="x", ipady=7, pady=(3, 14))

        row = tk.Frame(form, bg=C["bg"])
        row.pack(fill="x", pady=3)
        tk.Label(row, text="Sequences per sign", font=F(9),
                 bg=C["bg"], fg=C["muted"]).pack(side="left")
        self._cv_seqs = tk.StringVar(value="30")
        entry_field(row, self._cv_seqs).pack(side="right", ipady=6)

        row2 = tk.Frame(form, bg=C["bg"])
        row2.pack(fill="x", pady=3)
        tk.Label(row2, text="Frames per sequence", font=F(9),
                 bg=C["bg"], fg=C["muted"]).pack(side="left")
        self._cv_frames = tk.StringVar(value="30")
        entry_field(row2, self._cv_frames).pack(side="right", ipady=6)

        tk.Frame(form, bg=C["border"], height=1).pack(fill="x", pady=12)

        btn_row = tk.Frame(form, bg=C["bg"])
        btn_row.pack(fill="x")
        Btn(btn_row, "Start Collecting", self._start_collect,
            bg=C["success"]).pack(side="left", padx=(0, 8))
        Btn(btn_row, "Stop", self._stop_collect, bg=C["danger"]).pack(side="left")

        self._cv_prog_lbl = tk.Label(form, text="", font=F(8),
                                     bg=C["bg"], fg=C["muted"], anchor="w")
        self._cv_prog_lbl.pack(anchor="w", pady=(14, 3))
        self._cv_prog = ttk.Progressbar(form, mode="determinate")
        self._cv_prog.pack(fill="x")

        tk.Frame(form, bg=C["border"], height=1).pack(fill="x", pady=12)

        SectionLabel(form, "EXISTING CLASSES").pack(anchor="w", pady=(0, 6))
        self._cls_list_frame = tk.Frame(form, bg=C["bg"])
        self._cls_list_frame.pack(fill="x")

        pf = tk.Frame(body, bg=C["border"], bd=1)
        pf.pack(side="right", fill="both", expand=True)
        self._collect_canvas = tk.Canvas(pf, bg="#000", highlightthickness=0)
        self._collect_canvas.pack(fill="both", expand=True)

    def _refresh_class_list(self):
        for w in self._cls_list_frame.winfo_children():
            w.destroy()
        classes = detect_classes()
        if not classes:
            tk.Label(self._cls_list_frame, text="None yet",
                     font=F(9), bg=C["bg"], fg=C["muted"]).pack(anchor="w")
            return
        for c in classes:
            n = count_sequences(c)
            tk.Label(self._cls_list_frame,
                     text=f"  {c}   ({n} seqs)",
                     font=F(9), bg=C["bg"], fg=C["text"]).pack(anchor="w")

    def _start_collect(self):
        raw   = self._cv_signs.get()
        signs = [s.strip() for s in raw.split(",") if s.strip()]
        if not signs:
            messagebox.showwarning("No signs", "Enter sign names.")
            return
        seqs   = int(self._cv_seqs.get()   or SEQS_PER_SIGN)
        frames = int(self._cv_frames.get() or SEQ_LEN)
        src    = self._collect_cam.get_source()
        self._collect_th = CollectThread(
            signs, frames, seqs, self._collect_sq, self._collect_fq, cam_src=src)
        self._collect_th.start()

    def _stop_collect(self):
        if self._collect_th:
            self._collect_th.stop()
            self._collect_th = None
        self._refresh_class_list()

    # ==========================================================
    # PAGE 3 - TRAIN MODEL
    # ==========================================================
    def _build_train_page(self):
        p = tk.Frame(self._main, bg=C["bg"])
        self._pages["train"] = p

        self._page_header(p, "Train Model",
                          "LSTM 64->128->64->Dense — all gestures trained together from scratch")

        body = tk.Frame(p, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        left = tk.Frame(body, bg=C["bg"], width=320)
        left.pack(side="left", fill="y", padx=(0, 20))
        left.pack_propagate(False)

        # New gesture info banner
        info_box = tk.Frame(left, bg=C["surface2"], padx=10, pady=8)
        info_box.pack(fill="x", pady=(0, 14))
        tk.Label(info_box,
                 text="Adding a new gesture?",
                 font=F(9, True), bg=C["surface2"], fg=C["accent"],
                 anchor="w").pack(fill="x")
        tk.Label(info_box,
                 text=(
                     "1. Collect Data for the new gesture\n"
                     "2. Come back here and click Refresh\n"
                     "3. Click Start Training\n\n"
                     "Training always rebuilds model.h5 from\n"
                     "scratch using ALL gestures in MP_Data/.\n"
                     "Old model is overwritten automatically."
                 ),
                 font=F(8), bg=C["surface2"], fg=C["text"],
                 justify="left", anchor="w").pack(fill="x")

        SectionLabel(left, "CLASSES TO TRAIN (auto-detected from MP_Data/)").pack(anchor="w")
        self._tr_classes_lbl = tk.Label(left, text="-", font=F(10),
                                        bg=C["bg"], fg=C["text"], wraplength=300, justify="left")
        self._tr_classes_lbl.pack(anchor="w", pady=(3, 4))

        Btn(left, "↻ Refresh Class List", self._refresh_train_classes,
            small=True).pack(anchor="w", pady=(0, 14))

        row = tk.Frame(left, bg=C["bg"])
        row.pack(fill="x", pady=3)
        tk.Label(row, text="Epochs", font=F(9),
                 bg=C["bg"], fg=C["muted"]).pack(side="left")
        self._tr_epochs = tk.StringVar(value=str(DEFAULT_EPOCHS))
        entry_field(row, self._tr_epochs).pack(side="right", ipady=6)

        row2 = tk.Frame(left, bg=C["bg"])
        row2.pack(fill="x", pady=3)
        tk.Label(row2, text="Sequence length", font=F(9),
                 bg=C["bg"], fg=C["muted"]).pack(side="left")
        self._tr_seqlen = tk.StringVar(value=str(SEQ_LEN))
        entry_field(row2, self._tr_seqlen).pack(side="right", ipady=6)

        self._tr_gpu = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Use GPU (if available)",
                       variable=self._tr_gpu, font=F(9),
                       bg=C["bg"], fg=C["text"], selectcolor=C["surface2"],
                       activebackground=C["bg"], activeforeground=C["text"]).pack(anchor="w", pady=6)

        tk.Frame(left, bg=C["border"], height=1).pack(fill="x", pady=10)

        btn_row = tk.Frame(left, bg=C["bg"])
        btn_row.pack(fill="x")
        Btn(btn_row, "Start Training", self._start_train,
            bg=C["accent"], fg=C["bg"]).pack(side="left", padx=(0, 8))
        Btn(btn_row, "Stop", self._stop_train, bg=C["danger"]).pack(side="left")

        self._tr_prog_lbl = tk.Label(left, text="", font=F(8),
                                     bg=C["bg"], fg=C["muted"], anchor="w")
        self._tr_prog_lbl.pack(anchor="w", pady=(14, 3))
        self._tr_prog = ttk.Progressbar(left, mode="determinate")
        self._tr_prog.pack(fill="x", pady=(0, 10))

        SectionLabel(left, "LOG").pack(anchor="w", pady=(4, 3))
        self._tr_log = scrolledtext.ScrolledText(
            left, bg=C["surface2"], fg=C["text"], font=FM(8),
            bd=0, relief="flat", height=16, wrap="word",
            state="disabled")
        self._tr_log.pack(fill="both", expand=True)

        right = tk.Frame(body, bg=C["surface2"], bd=1)
        right.pack(side="right", fill="both", expand=True)
        tk.Label(right, text="Model summary will appear here after training starts",
                 font=F(9), bg=C["surface2"], fg=C["muted"],
                 wraplength=300).pack(expand=True)

    def _refresh_train_classes(self):
        classes = detect_classes()
        if classes:
            self._tr_classes_lbl.config(
                text=", ".join(classes) + f"  ({len(classes)} total)",
                fg=C["text"])
        else:
            self._tr_classes_lbl.config(
                text="No data found in MP_Data/ — collect data first",
                fg=C["danger"])

    def _start_train(self):
        # Always re-scan MP_Data so newly collected gestures are included
        classes = detect_classes()
        if not classes:
            messagebox.showwarning(
                "No data",
                "No gesture data found in MP_Data/.\n\n"
                "Go to Collect Data tab first and record sequences\n"
                "for each gesture you want to train.")
            return
        # Update the displayed class list so user can see exactly what's included
        self._tr_classes_lbl.config(
            text=", ".join(classes) + f"  ({len(classes)} total)",
            fg=C["text"])
        epochs = int(self._tr_epochs.get() or DEFAULT_EPOCHS)
        seqlen = int(self._tr_seqlen.get() or SEQ_LEN)
        self._tr_log_clear()
        self._tr_log_append(
            f"Training from scratch on {len(classes)} gesture(s): {', '.join(classes)}\n"
            f"All classes in MP_Data/ are included automatically.\n"
            f"model.h5 will be overwritten when training completes.\n"
        )
        self._tr_prog["value"] = 0
        self._train_th = TrainThread(
            classes, epochs, seqlen, self._train_q,
            use_gpu=self._tr_gpu.get())
        self._train_th.start()

    def _stop_train(self):
        if self._train_th:
            self._train_th.stop()

    def _tr_log_clear(self):
        self._tr_log.config(state="normal")
        self._tr_log.delete("1.0", "end")
        self._tr_log.config(state="disabled")

    def _tr_log_append(self, text):
        self._tr_log.config(state="normal")
        self._tr_log.insert("end", str(text) + "\n")
        self._tr_log.see("end")
        self._tr_log.config(state="disabled")

    # ==========================================================
    # PAGE 4 - LIVE DETECT
    # ==========================================================
    def _build_detect_page(self):
        p = tk.Frame(self._main, bg=C["bg"])
        self._pages["detect"] = p

        self._page_header(p, "Live Detection",
                          "Real-time sign recognition + sentence builder")

        # Camera selector
        cam_row = tk.Frame(p, bg=C["bg"])
        cam_row.pack(fill="x", padx=24, pady=(0, 8))
        self._detect_cam = CameraSelector(
            cam_row,
            initial=str(self.settings.get("camera_source", "0")))
        self._detect_cam.pack(side="left")

        body = tk.Frame(p, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=(0, 0))

        vf = tk.Frame(body, bg=C["border"])
        vf.pack(side="left", fill="both", expand=True)
        self._detect_canvas = tk.Canvas(vf, bg="#000", highlightthickness=0)
        self._detect_canvas.pack(fill="both", expand=True)

        rp = tk.Frame(body, bg=C["bg"], width=280)
        rp.pack(side="right", fill="y", padx=(16, 0))
        rp.pack_propagate(False)

        SectionLabel(rp, "CURRENT SIGN").pack(anchor="w")
        self._dt_word = tk.Label(rp, text="-",
                                 font=F(34, True), bg=C["bg"], fg=C["accent"])
        self._dt_word.pack(anchor="w")
        self._dt_conf = tk.Label(rp, text="",
                                 font=F(9), bg=C["bg"], fg=C["muted"])
        self._dt_conf.pack(anchor="w", pady=(0, 12))

        SectionLabel(rp, "PROBABILITIES").pack(anchor="w")
        self._dt_prob_frame = tk.Frame(rp, bg=C["bg"])
        self._dt_prob_frame.pack(fill="x", pady=(4, 14))
        self._dt_prob_signs = ()
        self._dt_prob_widgets = {}

        tk.Frame(rp, bg=C["border"], height=1).pack(fill="x", pady=6)

        SectionLabel(rp, "SENTENCE").pack(anchor="w")
        sb = tk.Frame(rp, bg=C["surface2"], padx=10, pady=8)
        sb.pack(fill="x", pady=(4, 8))
        self._dt_sent = tk.Label(sb, text="",
                                 font=F(12), bg=C["surface2"], fg=C["text"],
                                 wraplength=240, justify="left", anchor="w", height=2)
        self._dt_sent.pack(fill="x")

        sc = tk.Frame(rp, bg=C["bg"])
        sc.pack(fill="x", pady=(0, 12))
        Btn(sc, "<- Undo",  self._detect_undo,  small=True).pack(side="left", padx=(0, 6))
        Btn(sc, "X Clear", self._detect_clear, small=True,
            bg=C["danger"]).pack(side="left")
        Btn(sc, "Copy Copy",  self._detect_copy,  small=True).pack(side="right")

        tk.Frame(rp, bg=C["border"], height=1).pack(fill="x", pady=6)

        SectionLabel(rp, "CONTROLS").pack(anchor="w", pady=(0, 6))

        self._dt_thresh_var = tk.StringVar(
            value=str(self.settings.get("threshold", DEFAULT_THRESH)))
        row = tk.Frame(rp, bg=C["bg"])
        row.pack(fill="x", pady=3)
        tk.Label(row, text="Threshold", font=F(9),
                 bg=C["bg"], fg=C["muted"]).pack(side="left")
        entry_field(row, self._dt_thresh_var).pack(side="right", ipady=5)

        self._dt_probviz_var = tk.BooleanVar(value=True)
        tk.Checkbutton(rp, text="Show prob_viz overlay",
                       variable=self._dt_probviz_var, font=F(9),
                       bg=C["bg"], fg=C["text"], selectcolor=C["surface2"],
                       activebackground=C["bg"], activeforeground=C["text"]).pack(anchor="w", pady=4)

        btn_row = tk.Frame(rp, bg=C["bg"])
        btn_row.pack(fill="x", pady=(6, 0))
        Btn(btn_row, "Start", self._start_detect,
            bg=C["success"]).pack(side="left", padx=(0, 8))
        Btn(btn_row, "Stop",  self._stop_detect,
            bg=C["danger"]).pack(side="left")

        self._detect_builder = None

    def _start_detect(self):
        if self.model is None:
            self._pending_detect_start = True
            if self._auto_load_model():
                return
            self._pending_detect_start = False
            messagebox.showwarning("No model", "Load or train a model first.")
            return
        self._stop_detect()
        thr = float(self._dt_thresh_var.get() or DEFAULT_THRESH)
        viz = self._dt_probviz_var.get()
        src = self._detect_cam.get_source()
        self._detect_th = DetectThread(
            self.model, self.signs, src, self._detect_q, thr, viz)
        self._detect_th.start()
        self._draw_prob_bars(self._dt_prob_frame, [0.0] * len(self.signs), self.signs)

    def _stop_detect(self):
        if self._detect_th:
            self._detect_th.stop()
            self._detect_th = None

    def _detect_undo(self):
        if self._detect_builder:
            self._detect_builder.undo()
            self._dt_sent.config(text=self._detect_builder.text())

    def _detect_clear(self):
        if self._detect_builder:
            self._detect_builder.clear()
            self._dt_sent.config(text="")

    def _detect_copy(self):
        self.clipboard_clear()
        self.clipboard_append(self._dt_sent.cget("text"))

    # ==========================================================
    # PAGE 5 - VIDEO FILE DETECT
    # ==========================================================
    def _build_video_page(self):
        p = tk.Frame(self._main, bg=C["bg"])
        self._pages["video"] = p

        self._page_header(p, "Video File Detection",
                          "Run inference on a pre-recorded video file")

        top = tk.Frame(p, bg=C["bg"])
        top.pack(fill="x", padx=24, pady=(0, 12))

        self._vd_path_var = tk.StringVar(value="")
        entry_field(top, self._vd_path_var, wide=True).pack(
            side="left", fill="x", expand=True, ipady=7, padx=(0, 8))
        Btn(top, "Browse...", self._browse_video).pack(side="left", padx=(0, 8))
        Btn(top, "Run", self._run_video_detect,
            bg=C["success"]).pack(side="left", padx=(0, 8))
        Btn(top, "Stop", self._stop_detect, bg=C["danger"]).pack(side="left")

        body = tk.Frame(p, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        vf = tk.Frame(body, bg=C["border"])
        vf.pack(side="left", fill="both", expand=True)
        self._video_canvas = tk.Canvas(vf, bg="#000", highlightthickness=0)
        self._video_canvas.pack(fill="both", expand=True)

        rp = tk.Frame(body, bg=C["bg"], width=260)
        rp.pack(side="right", fill="y", padx=(16, 0))
        rp.pack_propagate(False)

        SectionLabel(rp, "SENTENCE").pack(anchor="w")
        sb = tk.Frame(rp, bg=C["surface2"], padx=10, pady=8)
        sb.pack(fill="x", pady=(4, 8))
        self._vd_sent = tk.Label(sb, text="",
                                 font=F(12), bg=C["surface2"], fg=C["text"],
                                 wraplength=220, justify="left", anchor="w")
        self._vd_sent.pack(fill="x")

        SectionLabel(rp, "CURRENT SIGN").pack(anchor="w", pady=(10, 0))
        self._vd_word = tk.Label(rp, text="-",
                                 font=F(28, True), bg=C["bg"], fg=C["accent"])
        self._vd_word.pack(anchor="w")

    def _browse_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*")])
        if path:
            self._vd_path_var.set(path)

    def _run_video_detect(self):
        if self.model is None:
            messagebox.showwarning("No model", "Load or train a model first.")
            return
        path = self._vd_path_var.get().strip()
        if not path or not Path(path).exists():
            messagebox.showwarning("No file", "Select a valid video file.")
            return
        self._stop_detect()
        self._detect_th = DetectThread(
            self.model, self.signs, path, self._detect_q,
            show_prob_viz=True)
        self._detect_th.start()

    # ==========================================================
    # PAGE 6 - EVALUATION
    # ==========================================================
    def _build_eval_page(self):
        p = tk.Frame(self._main, bg=C["bg"])
        self._pages["eval"] = p

        self._page_header(p, "Model Evaluation",
                          "Confusion matrix + accuracy on held-out test split (5%)")

        body = tk.Frame(p, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        left = tk.Frame(body, bg=C["bg"], width=320)
        left.pack(side="left", fill="y", padx=(0, 20))
        left.pack_propagate(False)

        SectionLabel(left, "CLASSES TO EVALUATE").pack(anchor="w")
        self._ev_classes_lbl = tk.Label(left, text="-", font=F(10),
                                        bg=C["bg"], fg=C["text"], wraplength=300, justify="left")
        self._ev_classes_lbl.pack(anchor="w", pady=(3, 14))

        row = tk.Frame(left, bg=C["bg"])
        row.pack(fill="x", pady=3)
        tk.Label(row, text="Test split", font=F(9),
                 bg=C["bg"], fg=C["muted"]).pack(side="left")
        self._ev_split = tk.StringVar(value="0.05")
        entry_field(row, self._ev_split).pack(side="right", ipady=6)

        tk.Frame(left, bg=C["border"], height=1).pack(fill="x", pady=10)
        Btn(left, "Run Evaluation", self._run_eval,
            bg=C["accent"], fg=C["bg"]).pack(anchor="w")

        self._ev_acc_lbl = tk.Label(left, text="", font=F(20, True),
                                    bg=C["bg"], fg=C["success"])
        self._ev_acc_lbl.pack(anchor="w", pady=(16, 0))

        right = tk.Frame(body, bg=C["surface2"])
        right.pack(side="right", fill="both", expand=True)
        SectionLabel(right, "RESULTS").pack(anchor="w", padx=12, pady=(10, 4))
        self._ev_log = scrolledtext.ScrolledText(
            right, bg=C["surface2"], fg=C["text"], font=FM(9),
            bd=0, relief="flat", wrap="word", state="disabled")
        self._ev_log.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _refresh_eval_classes(self):
        classes = detect_classes()
        self._ev_classes_lbl.config(
            text=", ".join(classes) if classes else "No data")

    def _run_eval(self):
        if self.model is None:
            messagebox.showwarning("No model", "Load a model first.")
            return
        classes = detect_classes()
        if not classes:
            messagebox.showwarning("No data", "Collect data first.")
            return
        split = float(self._ev_split.get() or 0.05)

        def _do():
            try:
                import tensorflow as tf
                from tensorflow.keras.utils import to_categorical
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

                label_map  = {s: i for i, s in enumerate(classes)}
                sequences, labels = [], []
                for sign in classes:
                    sign_dir = Path(DATA_PATH, sign)
                    try:
                        seq_ids = sorted(int(x) for x in os.listdir(sign_dir)
                                         if Path(sign_dir, x).is_dir())
                    except Exception:
                        continue
                    for seq in seq_ids:
                        window = []
                        for f in range(SEQ_LEN):
                            fp = Path(DATA_PATH, sign, str(seq), f"{f}.npy")
                            if fp.exists():
                                window.append(np.load(fp))
                        if len(window) == SEQ_LEN:
                            sequences.append(window)
                            labels.append(label_map[sign])

                X = np.array(sequences)
                y = to_categorical(labels).astype(int)
                _, X_test, _, y_test = train_test_split(X, y, test_size=split)

                yhat_raw = self.model.predict(X_test, verbose=0)
                ytrue    = np.argmax(y_test, axis=1).tolist()
                yhat     = np.argmax(yhat_raw, axis=1).tolist()
                acc      = accuracy_score(ytrue, yhat)
                cm       = multilabel_confusion_matrix(ytrue, yhat)

                self.after(0, lambda: self._ev_show(acc, cm, classes, ytrue, yhat))
            except Exception as e:
                import traceback
                self.after(0, lambda: self._ev_log_set(traceback.format_exc()))

        threading.Thread(target=_do, daemon=True).start()

    def _ev_show(self, acc, cm, classes, ytrue, yhat):
        self._ev_acc_lbl.config(text=f"{acc:.1%}")
        lines = [f"Test accuracy: {acc:.4f} ({acc:.1%})\n",
                 f"Samples tested: {len(ytrue)}\n",
                 "\nConfusion matrices (one per class):"]
        for i, c in enumerate(classes):
            lines.append(f"\n  {c}:\n{cm[i]}")
        self._ev_log_set("\n".join(lines))

    def _ev_log_set(self, text):
        self._ev_log.config(state="normal")
        self._ev_log.delete("1.0", "end")
        self._ev_log.insert("end", text)
        self._ev_log.config(state="disabled")

    # ==========================================================
    # PAGE 7 - SETTINGS
    # ==========================================================
    def _build_settings_page(self):
        p = tk.Frame(self._main, bg=C["bg"])
        self._pages["settings"] = p

        self._page_header(p, "Settings", "")

        body = tk.Frame(p, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        left = tk.Frame(body, bg=C["bg"], width=460)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        def labeled_entry(parent, label, var, wide=False):
            row = tk.Frame(parent, bg=C["bg"])
            row.pack(fill="x", pady=5)
            tk.Label(row, text=label, font=F(9), bg=C["bg"],
                     fg=C["muted"], width=28, anchor="w").pack(side="left")
            entry_field(row, var, wide=wide).pack(
                side="right", fill="x" if wide else "none",
                expand=wide, ipady=6, padx=(8, 0))

        # -- Camera --------------------------------------------
        SectionLabel(left, "DEFAULT CAMERA SOURCE").pack(anchor="w", pady=(0, 4))
        cam_help = tk.Label(
            left,
            text=(
                "Enter a camera index (0, 1, 2...) or a full URL.\n"
                "DroidCam (USB): usually index 1 or 2.\n"
                "DroidCam (WiFi): http://<phone-ip>:4747/video\n"
                "  e.g. http://192.168.1.5:4747/video"
            ),
            font=F(8), bg=C["bg"], fg=C["muted"], justify="left", anchor="w")
        cam_help.pack(anchor="w", pady=(0, 6))

        self._st_cam = tk.StringVar(
            value=str(self.settings.get("camera_source", "0")))
        cam_entry_row = tk.Frame(left, bg=C["bg"])
        cam_entry_row.pack(fill="x", pady=3)
        tk.Label(cam_entry_row, text="Camera source", font=F(9),
                 bg=C["bg"], fg=C["muted"], width=28, anchor="w").pack(side="left")
        entry_field(cam_entry_row, self._st_cam, wide=True).pack(
            side="right", fill="x", expand=True, ipady=6, padx=(8, 0))

        tk.Frame(left, bg=C["border"], height=1).pack(fill="x", pady=12)

        # -- Inference -----------------------------------------
        SectionLabel(left, "INFERENCE").pack(anchor="w", pady=(0, 6))
        self._st_thresh = tk.StringVar(
            value=str(self.settings.get("threshold", DEFAULT_THRESH)))
        labeled_entry(left, "Confidence threshold", self._st_thresh)

        tk.Frame(left, bg=C["border"], height=1).pack(fill="x", pady=12)
        SectionLabel(left, "PATHS").pack(anchor="w", pady=(0, 6))
        self._st_model = tk.StringVar(
            value=self.settings.get("model_path", MODEL_PATH))
        self._st_data  = tk.StringVar(
            value=self.settings.get("data_path",  DATA_PATH))
        labeled_entry(left, "Model path (.h5)",  self._st_model, wide=True)
        labeled_entry(left, "Data folder",        self._st_data,  wide=True)

        tk.Frame(left, bg=C["border"], height=1).pack(fill="x", pady=12)
        SectionLabel(left, "GPU").pack(anchor="w", pady=(0, 6))
        self._st_gpu = tk.BooleanVar(
            value=self.settings.get("use_gpu", True))
        tk.Checkbutton(left, text="Enable GPU for training",
                       variable=self._st_gpu, font=F(9),
                       bg=C["bg"], fg=C["text"], selectcolor=C["surface2"],
                       activebackground=C["bg"], activeforeground=C["text"]).pack(anchor="w")

        tk.Frame(left, bg=C["border"], height=1).pack(fill="x", pady=12)

        btn_row = tk.Frame(left, bg=C["bg"])
        btn_row.pack(fill="x")
        Btn(btn_row, "Save Settings", self._save_settings,
            bg=C["accent"], fg=C["bg"]).pack(side="left", padx=(0, 8))
        Btn(btn_row, "Browse & Load Model...",
            self._browse_load_model).pack(side="left")

    def _save_settings(self):
        self.settings["threshold"]     = float(self._st_thresh.get())
        self.settings["model_path"]    = self._st_model.get()
        self.settings["data_path"]     = self._st_data.get()
        self.settings["use_gpu"]       = self._st_gpu.get()
        self.settings["camera_source"] = self._st_cam.get()
        save_settings(self.settings)
        messagebox.showinfo("Saved", "Settings saved.")

    def _browse_load_model(self):
        path = filedialog.askopenfilename(
            title="Select model .h5",
            filetypes=[("HDF5", "*.h5"), ("All", "*")])
        if path:
            self._st_model.set(path)
            self._load_model(path)

    # ==========================================================
    # MODEL LOADING
    # ==========================================================
    def _auto_load_model(self):
        if self.model is not None or self._model_loading:
            return False

        candidates = []
        seen = set()
        configured = self.settings.get("model_path", MODEL_PATH)
        if configured and configured not in seen:
            candidates.append(configured)
            seen.add(configured)
        if MODEL_PATH not in seen:
            candidates.append(MODEL_PATH)
            seen.add(MODEL_PATH)
        for p in Path(".").glob("*.h5"):
            if p.name.endswith("_weights.h5"):
                continue
            sp = str(p)
            if sp not in seen:
                candidates.append(sp)
                seen.add(sp)

        for path in candidates:
            if Path(path).exists():
                self._load_model(path)
                return True

        classes = detect_classes(self.settings.get("data_path", DATA_PATH))
        if classes:
            self.signs = classes
            self._lbl_classes.config(text="  ".join(classes))
        return False

    def _load_model(self, path):
        if self._model_loading:
            return
        self._model_loading = True
        self._lbl_mstatus.config(text="Loading...", fg=C["muted"])

        def _do():
            try:
                import tensorflow as tf
                m  = tf.keras.models.load_model(path)
                wp = path.replace(".h5", "_weights.h5")
                if Path(wp).exists():
                    m.load_weights(wp)
                dp      = self.settings.get("data_path", DATA_PATH)
                classes = detect_classes(dp)
                if not classes:
                    n = m.output_shape[-1]
                    classes = [f"sign_{i}" for i in range(n)]
                self.after(0, lambda: self._on_loaded(m, classes, path))
            except Exception as e:
                self.after(0, lambda err=e: self._on_load_failed(err))

        threading.Thread(target=_do, daemon=True).start()

    def _on_loaded(self, model, classes, path):
        self._model_loading = False
        self.model = model
        self.signs = classes
        self._lbl_mstatus.config(text=f"OK {Path(path).name}", fg=C["success"])
        self._lbl_classes.config(text="  ".join(classes))
        if self._pending_detect_start:
            self._pending_detect_start = False
            self.after(0, self._start_detect)

    def _on_load_failed(self, err=None):
        self._model_loading = False
        was_pending_detect = self._pending_detect_start
        self._pending_detect_start = False
        msg = "FAIL Load failed"
        if err:
            detail = str(err).strip()
            if not detail:
                detail = "Load failed"
            if len(detail) > ERROR_MSG_MAX_LEN:
                detail = f"{detail[:ERROR_MSG_MAX_LEN]}..."
            msg = f"FAIL {err.__class__.__name__}: {detail}"
        self._lbl_mstatus.config(text=msg, fg=C["danger"])
        if was_pending_detect:
            messagebox.showwarning("No model", "Load or train a model first.")

    # ==========================================================
    # MAIN TICK - drains all queues every 40 ms
    # ==========================================================
    def _tick(self):
        # preview
        try:
            while True:
                frame = self._preview_q.get_nowait()
                self._photos["preview"] = show_frame_on_canvas(
                    self._preview_canvas, frame,
                    self._photos.get("preview"))
        except queue.Empty:
            pass

        # check preview thread for errors
        if self._preview_th and not self._preview_th.is_alive():
            err = getattr(self._preview_th, "error", None)
            if err:
                self._preview_err_lbl.config(text=f"FAIL {err}")
                self._preview_th = None

        # collect frames
        try:
            while True:
                frame = self._collect_fq.get_nowait()
                self._photos["collect"] = show_frame_on_canvas(
                    self._collect_canvas, frame,
                    self._photos.get("collect"))
        except queue.Empty:
            pass

        # collect status
        try:
            while True:
                s = self._collect_sq.get_nowait()
                if s.get("done_all"):
                    self._cv_prog["value"] = 100
                    self._cv_prog_lbl.config(text="OK Collection complete")
                    self._refresh_class_list()
                elif s.get("error"):
                    self._cv_prog_lbl.config(
                        text=f"FAIL {s['error']}", fg=C["danger"])
                else:
                    self._cv_prog["value"] = s.get("pct", 0)
                    self._cv_prog_lbl.config(
                        text=f"{s['sign']}  seq {s['seq']}  "
                             f"frame {s['frame']}/{SEQ_LEN}")
        except queue.Empty:
            pass

        # training
        try:
            while True:
                msg = self._train_q.get_nowait()
                if msg[0] == "log":
                    self._tr_log_append(msg[1])
                elif msg[0] == "progress":
                    self._tr_prog["value"] = msg[1]
                    self._tr_prog_lbl.config(text=msg[2])
                    self._tr_log_append(msg[2])
                elif msg[0] == "done":
                    self._tr_log_append(f"\nOK Complete - test accuracy: {msg[1]:.1%}")
                    self._tr_prog["value"] = 100
                    self._auto_load_model()
                elif msg[0] in ("error", "cancelled"):
                    self._tr_log_append(f"FAIL {msg[-1]}")
        except queue.Empty:
            pass

        # detection (live + video share same detect_q)
        try:
            while True:
                res = self._detect_q.get_nowait()
                if res.get("done"):
                    if res.get("error"):
                        messagebox.showerror("Camera error", res["error"])
                    break
                frame   = res["frame"]
                builder = res["builder"]
                self._detect_builder = builder

                if self._active == "detect":
                    self._photos["detect"] = show_frame_on_canvas(
                        self._detect_canvas, frame,
                        self._photos.get("detect"))
                    self._dt_word.config(text=(res["word"] or "-").upper())
                    conf = res["confidence"]
                    self._dt_conf.config(
                        text=f"{conf:.0%}" if conf else "",
                        fg=C["success"] if conf >= float(
                            self._dt_thresh_var.get() or DEFAULT_THRESH)
                        else C["muted"])
                    self._dt_sent.config(text=builder.text())
                    self._draw_prob_bars(
                        self._dt_prob_frame, res["probs"], res["signs"])

                elif self._active == "video":
                    self._photos["video"] = show_frame_on_canvas(
                        self._video_canvas, frame,
                        self._photos.get("video"))
                    self._vd_sent.config(text=builder.text())
                    self._vd_word.config(text=(res["word"] or "-").upper())
        except queue.Empty:
            pass

        self.after(40, self._tick)

    # ==========================================================
    # HELPERS
    # ==========================================================
    def _draw_prob_bars(self, container, probs, signs):
        normalized_signs = tuple(signs)
        if normalized_signs != self._dt_prob_signs:
            for w in container.winfo_children():
                w.destroy()
            self._dt_prob_widgets = {}
            self._dt_prob_signs = normalized_signs
            for s in normalized_signs:
                row = tk.Frame(container, bg=C["bg"])
                row.pack(fill="x", pady=1)
                tk.Label(row, text=s, font=F(8), bg=C["bg"],
                         fg=C["text"], width=14, anchor="w").pack(side="left")
                bg = tk.Frame(row, bg=C["border"], height=6)
                bg.pack(side="left", fill="x", expand=True, padx=(4, 0))
                bg.pack_propagate(False)
                fill = tk.Frame(bg, bg=C["accent_dk"], height=6)
                fill.place(x=0, y=0, relwidth=0.0)
                self._dt_prob_widgets[s] = fill

        probs = probs or []
        max_p = max(probs) if probs else 0
        for i, s in enumerate(self._dt_prob_signs):
            p = float(probs[i]) if i < len(probs) else 0.0
            p = max(0.0, min(1.0, p))
            fill = self._dt_prob_widgets.get(s)
            if fill is None:
                continue
            fill.configure(bg=C["accent"] if p >= max_p and p > 0 else C["accent_dk"])
            fill.place_configure(relwidth=p)

    def _page_header(self, parent, title, subtitle):
        hdr = tk.Frame(parent, bg=C["bg"])
        hdr.pack(fill="x", padx=24, pady=(20, 12))
        tk.Label(hdr, text=title, font=F(20, True),
                 bg=C["bg"], fg=C["text"]).pack(anchor="w")
        if subtitle:
            tk.Label(hdr, text=subtitle, font=F(9),
                     bg=C["bg"], fg=C["muted"]).pack(anchor="w", pady=(2, 0))
        tk.Frame(parent, bg=C["border"], height=1).pack(
            fill="x", padx=24, pady=(0, 12))
        return hdr

    def on_close(self):
        self._stop_preview()
        self._stop_detect()
        if self._collect_th:
            self._collect_th.stop()
        if self._train_th:
            self._train_th.stop()
        self.destroy()


# -----------------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

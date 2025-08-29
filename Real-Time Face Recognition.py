import os
import cv2
import time
import glob
import shutil
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

from skimage import exposure
from skimage.feature import hog, local_binary_pattern


# ---------------------------- Config & Utils ---------------------------- #
@dataclass
class Config:
    face_size: tuple[int, int] = (128, 128)
    lbp_P: int = 8
    lbp_R: int = 1
    lbp_method: str = "uniform"
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (8, 8)
    hog_cells_per_block: tuple[int, int] = (2, 2)
    detection_min_size: tuple[int, int] = (60, 60)
    predict_threshold: float = 0.60
    # Data augmentation
    augment_flip: bool = True
    augment_gamma: tuple[float, float] = (0.8, 1.2)
    augment_rotation: tuple[int, int] = (-5, 5)


def get_face_detector() -> cv2.CascadeClassifier:
    """Loads and returns the pre-trained face cascade classifier."""
    return cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))


def detect_faces_bgr(bgr: np.ndarray, min_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Detects faces in a BGR image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = get_face_detector().detectMultiScale(gray, 1.1, 5, minSize=min_size)
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def crop(bgr: np.ndarray, rect: tuple[int, int, int, int], out_size: tuple[int, int]) -> np.ndarray:
    """Crops and resizes a region from an image."""
    x, y, w, h = rect
    chip = bgr[max(0, y):y + h, max(0, x):x + w]
    return cv2.resize(chip, out_size)


def _gamma(img: np.ndarray, g: float) -> np.ndarray:
    """Applies gamma correction to an image."""
    if g == 1.0:
        return img
    img_f = img.astype(np.float32) / 255.0
    out = np.power(np.clip(img_f, 0, 1), g)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotates an image."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def preprocess(gray_or_bgr: np.ndarray) -> np.ndarray:
    """Preprocesses a face image by converting to grayscale and applying CLAHE."""
    gray = gray_or_bgr
    if gray_or_bgr.ndim == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    clahe = exposure.equalize_adapthist(gray, clip_limit=0.03)
    return (clahe * 255.0).astype(np.uint8)


def lbp_hist(gray: np.ndarray, P: int, R: int, method: str) -> np.ndarray:
    """Computes Local Binary Pattern (LBP) histogram."""
    lbp = local_binary_pattern(gray, P=P, R=R, method=method)
    n_bins = P + 3 if method == 'uniform' else int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def features(gray: np.ndarray, cfg: Config) -> np.ndarray:
    """Extracts HOG and LBP features from a grayscale image."""
    hog_feat = hog(gray, orientations=cfg.hog_orientations,
                   pixels_per_cell=cfg.hog_pixels_per_cell,
                   cells_per_block=cfg.hog_cells_per_block,
                   block_norm='L2-Hys', transform_sqrt=True, feature_vector=True).astype(np.float32)
    return np.hstack([hog_feat, lbp_hist(gray, cfg.lbp_P, cfg.lbp_R, cfg.lbp_method)])


# ---------------------------- Data IO ---------------------------- #

def ensure_dir(p: str) -> None:
    """Ensures a directory exists."""
    os.makedirs(p, exist_ok=True)


def copy_images_to_person(dst_root: str, person: str, paths: list[str]) -> int:
    """Copies image files to a person's dataset folder."""
    person_dir = os.path.join(dst_root, person)
    ensure_dir(person_dir)
    count = 0
    for src in paths:
        if not os.path.isfile(src):
            continue
        ext = os.path.splitext(src)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            continue
        base = os.path.basename(src)
        target = os.path.join(person_dir, base)
        i = 1
        while os.path.exists(target):
            name, ext2 = os.path.splitext(base)
            target = os.path.join(person_dir, f"{name}_{i}{ext2}")
            i += 1
        shutil.copy2(src, target)
        count += 1
    return count


def save_current_face(dst_root: str, person: str, frame_bgr: np.ndarray, rect: tuple[int, int, int, int], cfg: Config) -> str:
    """Saves a detected face chip from the webcam to the dataset."""
    person_dir = os.path.join(dst_root, person)
    ensure_dir(person_dir)
    chip = crop(frame_bgr, rect, cfg.face_size)
    fname = os.path.join(person_dir, f"webcam_{int(time.time())}.jpg")
    cv2.imwrite(fname, chip)
    return fname


def iter_dataset(root: str) -> list[tuple[str, str]]:
    """Iterates through the dataset directory and returns file-label pairs."""
    pairs = []
    for person_dir in sorted([d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d)]):
        label = os.path.basename(person_dir)
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'):
            for f in glob.glob(os.path.join(person_dir, ext)):
                pairs.append((f, label))
    return pairs


# ---------------------------- Model ---------------------------- #
class PackedModel:
    """A class to pack and save the trained model pipeline and label encoder."""
    def __init__(self, pipe: Pipeline, le: LabelEncoder, cfg: Config):
        self.pipe = pipe
        self.le = le
        self.cfg = cfg

    def save(self, path: str) -> None:
        """Saves the model to a file."""
        joblib.dump({'pipe': self.pipe, 'le': self.le, 'cfg': self.cfg}, path)

    @staticmethod
    def load(path: str) -> 'PackedModel':
        """Loads the model from a file."""
        obj = joblib.load(path)
        return PackedModel(pipe=obj['pipe'], le=obj['le'], cfg=obj['cfg'])


def _maybe_augment(gray: np.ndarray, cfg: Config) -> list[np.ndarray]:
    """Applies data augmentation techniques (gamma, rotation, flip)."""
    outs = [gray]

    g0, g1 = cfg.augment_gamma
    g = np.random.uniform(g0, g1)
    outs.append(_gamma(gray, g))

    r0, r1 = cfg.augment_rotation
    r = np.random.uniform(r0, r1)
    outs.append(_rotate(gray, r))

    if cfg.augment_flip:
        flipped_outs = [cv2.flip(img, 1) for img in outs]
        outs.extend(flipped_outs)

    return outs


def build_pipe() -> Pipeline:
    """Builds the machine learning pipeline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, svd_solver='full', random_state=42)),
        ('svc', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, class_weight='balanced', random_state=42))
    ])


def train_model(dataset_dir: str, cfg: Config) -> PackedModel:
    """Trains the face recognition model."""
    pairs = iter_dataset(dataset_dir)
    if not pairs:
        raise RuntimeError('Dataset is empty. Add some images first.')

    X, y = [], []
    for fp, label in pairs:
        img = cv2.imread(fp)
        if img is None:
            continue
        faces = detect_faces_bgr(img, cfg.detection_min_size)
        if not faces:
            continue
        rect = max(faces, key=lambda r: r[2]*r[3])
        chip = crop(img, rect, cfg.face_size)
        gray = preprocess(chip)

        augmented_chips = _maybe_augment(gray, cfg)
        for aug_chip in augmented_chips:
            X.append(features(aug_chip, cfg))
            y.append(label)

    if not X:
        raise RuntimeError('No faces extracted from dataset images.')

    X = np.vstack(X)
    y = np.array(y)
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)

    pipe = build_pipe()
    pipe.fit(X, y_enc)
    acc = accuracy_score(y_enc, pipe.predict(X))
    print(f"[train] In-sample accuracy: {acc:.3f}")

    return PackedModel(pipe, le, cfg)


def predict_face(gray: np.ndarray, pm: PackedModel) -> tuple[str, float]:
    """Predicts the identity of a face."""
    f = features(gray, pm.cfg).reshape(1, -1)

    probas = pm.pipe.predict_proba(f)[0]
    idx = np.argmax(probas)
    conf = probas[idx]

    name = pm.le.inverse_transform([idx])[0]

    return ("Unknown", conf) if conf < pm.cfg.predict_threshold else (name, conf)


def draw_overlay(frame: np.ndarray, rect: tuple[int, int, int, int], name: str, conf: Optional[float]) -> None:
    """Draws a bounding box and text on the frame."""
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 2
    name_line = f"Name: {name}"
    conf_line = f"Confidence: {conf:.6f}" if conf is not None else None
    (tw1, th1), _ = cv2.getTextSize(name_line, font, scale, thick)
    (tw2, th2), _ = cv2.getTextSize(conf_line or ' ', font, scale, thick)
    tw = max(tw1, tw2)
    total_h = th1 + (th2 + 6 if conf_line else 0)
    y_text_top = y - 12
    if y_text_top - total_h - 6 < 0:
        y_text_top = y + h + total_h + 12
    cv2.rectangle(frame, (x, y_text_top - total_h - 6), (x + tw + 10, y_text_top + 4), (0, 0, 0), -1)
    cv2.putText(frame, name_line, (x + 5, y_text_top - (th2 + 6 if conf_line else 0)), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    if conf_line:
        cv2.putText(frame, conf_line, (x + 5, y_text_top), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


# ---------------------------- GUI ---------------------------- #
class App:
    def __init__(self, root, dataset_dir: str, model_path: str, source: int, cfg: Config):
        self.root = root
        self.dataset_dir = dataset_dir
        self.model_path = model_path
        self.source = source
        self.cfg = cfg

        ensure_dir(dataset_dir)

        self.pm: Optional[PackedModel] = None
        if os.path.exists(model_path):
            try:
                self.pm = PackedModel.load(model_path)
            except Exception:
                self.pm = None

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            messagebox.showerror('Camera', f'Cannot open source {self.source}')
            raise SystemExit

        self.frame_bgr: Optional[np.ndarray] = None
        self._stop = False

        self.root.title('Face Recognition — Live (Optimized SVM)')
        self.root.bind('<KeyPress-q>', lambda e: self.on_close())

        main = ttk.Frame(root)
        main.pack(fill='both', expand=True)

        self.video_lbl = ttk.Label(main)
        self.video_lbl.grid(row=0, column=0, padx=8, pady=8, sticky='nsew')

        side = ttk.Frame(main)
        side.grid(row=0, column=1, padx=8, pady=8, sticky='n')

        ttk.Label(side, text='Full Name :').pack(anchor='w')
        self.name_var = tk.StringVar()
        ttk.Entry(side, textvariable=self.name_var, width=28).pack(anchor='w', pady=(0, 8))

        ttk.Button(side, text='Add Images…', command=self.add_images).pack(fill='x')
        ttk.Button(side, text='Capture From Webcam', command=self.capture_from_webcam).pack(fill='x', pady=4)
        ttk.Button(side, text='Train / Update Model', command=self.train_async).pack(fill='x')

        ttk.Separator(side, orient='horizontal').pack(fill='x', pady=8)

        self.thr_var = tk.DoubleVar(value=self.cfg.predict_threshold)
        ttk.Label(side, text='Unknown Threshold:').pack(anchor='w')
        thr = ttk.Scale(side, from_=0.4, to=0.95, variable=self.thr_var, command=self.on_threshold)
        thr.pack(fill='x')

        self.status_var = tk.StringVar(value='Ready')
        ttk.Label(side, textvariable=self.status_var, foreground='#007700').pack(anchor='w', pady=(8,0))

        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.update_loop()

    def add_images(self):
        person = self.name_var.get().strip()
        if not person:
            messagebox.showwarning('Name', 'Enter a full name first.')
            return
        paths = filedialog.askopenfilenames(title='Select images', filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')])
        if not paths:
            return
        n = copy_images_to_person(self.dataset_dir, person, list(paths))
        messagebox.showinfo('Add Images', f'Copied {n} image(s) to dataset/{person}. Click Train to update the model.')

    def capture_from_webcam(self):
        person = self.name_var.get().strip()
        if not person:
            messagebox.showwarning('Name', 'Enter a full name first.')
            return
        if self.frame_bgr is None:
            return
        faces = detect_faces_bgr(self.frame_bgr, self.cfg.detection_min_size)
        if not faces:
            messagebox.showwarning('Capture', 'No face detected in current frame.')
            return
        rect = max(faces, key=lambda r: r[2]*r[3])
        path = save_current_face(self.dataset_dir, person, self.frame_bgr, rect, self.cfg)
        messagebox.showinfo('Capture', f'Saved sample: {os.path.relpath(path)}\nClick Train to update the model.')

    def on_threshold(self, _=None):
        v = float(self.thr_var.get())
        self.cfg.predict_threshold = v
        if self.pm:
            self.pm.cfg.predict_threshold = v

    def train_async(self):
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self):
        try:
            self.status_var.set('Training…')
            pm = train_model(self.dataset_dir, self.cfg)
            pm.save(self.model_path)
            self.pm = pm
            self.status_var.set('Model trained ✓')
        except Exception as e:
            self.status_var.set('Train error')
            messagebox.showerror('Train', str(e))

    def update_loop(self):
        ok, frame = self.cap.read()
        if ok:
            self.frame_bgr = frame
            disp = frame.copy()
            if self.pm is not None:
                faces = detect_faces_bgr(disp, self.cfg.detection_min_size)
                for rect in faces:
                    chip = crop(disp, rect, self.cfg.face_size)
                    gray = preprocess(chip)
                    name, conf = predict_face(gray, self.pm)
                    draw_overlay(disp, rect, name, conf)
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_lbl.imgtk = imgtk
            self.video_lbl.configure(image=imgtk)
        if not self._stop:
            self.root.after(15, self.update_loop)

    def on_close(self):
        self._stop = True
        try:
            self.cap.release()
        except Exception:
            pass
        self.root.destroy()

def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass

    cfg = Config()


    dataset_dir = 'dataset'
    model_path = 'face_svm.joblib'
    source = 0


    app = App(root, dataset_dir=dataset_dir, model_path=model_path, source=source, cfg=cfg)
    root.protocol('WM_DELETE_WINDOW', app.on_close)
    root.mainloop()

if __name__ == '__main__':
    main()
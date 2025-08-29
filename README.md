Simple Multi-Face Recognition GUI (Tkinter + OpenCV + scikit-learn + scikit-image)
Optimized for speed. 

- Live webcam preview
- Right panel: enter Full Name 
  - Add Images…
  - Capture From Webcam
  - Train / Update Model
- On live frame, "Name: …" and "Confidence: …" are displayed for each face.

Install:
  pip install opencv-python scikit-image scikit-learn joblib numpy pillow

Run:
  python face_recognition_gui_optimized_svm.py --dataset dataset --model face_svm.joblib

Notes:
- After adding new samples (from file or webcam), click Train/Update to retrain the model.
- If you enter a new name, the folder dataset/<Full Name>/ will be created automatically.

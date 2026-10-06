# HandSignPlay

Play Hill Climb Racing hands-free: a webcam hand-gesture classifier (MediaPipe + Random Forest) that turns gestures into keyboard presses.

## What it does
The webcam stream is passed through MediaPipe Hands to extract 21 hand landmarks. A scikit-learn RandomForest classifier, trained on gestures you record yourself, predicts one of three classes, and `pyautogui` converts the prediction into held arrow keys so a desktop game (Hill Climb Racing, per the author) can be controlled without a keyboard.

## Gesture mapping (from `inference_classifier.py`)
| Class | Label shown | Key action |
|---|---|---|
| 0 | Rock | release all keys |
| 1 | Boost | hold Right arrow |
| 2 | Stop | hold Left arrow |

Note: the on-screen labels do not match the actions exactly (class 2 is labelled "Stop" but presses Left, class 1 is "Boost" but presses Right) - this follows the code as written.

## Pipeline
```
collect_imgs.py -> create_dataset.py -> train_classifier.py -> inference_classifier.py
(webcam jpgs)     (landmarks -> data.pickle) (RandomForest -> model.p) (live webcam + pyautogui)
```
1. `collect_imgs.py`: captures 100 webcam frames for each of 3 classes into `./data/<class>/`.
2. `create_dataset.py`: runs MediaPipe Hands on each image, normalises the 21 (x, y) landmark coordinates by subtracting the minimum x/y, and pickles features and labels.
3. `train_classifier.py`: 80/20 stratified split, `RandomForestClassifier`, prints accuracy, saves `model.p`.
4. `inference_classifier.py`: live loop; draws landmarks and a bounding box, predicts, and sends key presses.

## Tech stack
Python, OpenCV (`opencv-python==4.7.0.68`), MediaPipe, scikit-learn, pyautogui.

## Project structure
```
collect_imgs.py          # data capture
create_dataset.py        # landmark feature extraction
train_classifier.py      # training
inference_classifier.py  # real-time control
data.pickle, model.p     # pre-built features and trained model
data.zip                 # captured image dataset (about 14 MB)
requirements.txt
```

## Setup and run
```bash
pip install -r requirements.txt
python collect_imgs.py        # optional: record your own gestures (press Q to start each class)
python create_dataset.py
python train_classifier.py
python inference_classifier.py   # then focus the game window
```
Needs a webcam and a desktop session (pyautogui sends real key events).

## Limitations / future work
- Only 3 gestures; no reported accuracy in the repo (the training script prints it at run time).
- No debouncing/smoothing; `inference_classifier.py` has no quit key (loop runs until interrupted).
- Matplotlib is imported by `create_dataset.py` but is not in `requirements.txt`; numpy is only an implicit dependency.
- Labels vs. key mapping noted above should be harmonised.

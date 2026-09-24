# Multi-modal late fusion with multi-scale detection and temporal tracking.

This project contains a YOLOv8 person-detection training pipeline and a local rescue dashboard for viewing live detections.

## Project structure

- `train_person_yolo.py` — prepares the dataset split and trains the YOLOv8 model
- `live_rescue_web_app.py` — runs the local web dashboard for rescue locations
- `archive/person-3/` — the prepared YOLO dataset
- `yolov8n.pt` — pretrained YOLOv8 Nano weights
- `runs/train/` — training output and checkpoints
- `survivor_detection_result.csv` — output used by the rescue dashboard

## Requirements

- Python 3.10 or 3.11
- Windows 10/11 recommended
- A working internet connection for the first package install

## 1) Create and activate a Python environment

### Option A: venv

```powershell
cd C:\Users\admin\Downloads\megathon
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Option B: Conda

```powershell
conda create -n megathon python=3.11 -y
conda activate megathon
```

## 2) Install dependencies

From the project root:

```powershell
pip install --upgrade pip
pip install ultralytics torch opencv-python pandas numpy
```

If the environment is missing a dependency during training, install it explicitly:

```powershell
pip install ultralytics
```

## 3) Prepare the dataset

The person dataset is already stored in `archive/person-3`.

The training script automatically:

- checks for valid image/label pairs
- creates a validation split
- rewrites `archive/person-3/data.yaml`
- starts YOLOv8 training

Run:

```powershell
python train_person_yolo.py
```

This will create a validation set and train the model using `yolov8n.pt`.

### Notes

- Training output goes to `runs/train/person_detection/`
- The dataset config used by YOLO is `archive/person-3/data.yaml`
- The script uses a reproducible random seed for validation splitting

## 4) Run the rescue dashboard

From the project root:

```powershell
python live_rescue_web_app.py
```

Then open:

```text
http://127.0.0.1:8765
```

The app reads survivor data from `survivor_detection_result.csv` and serves the dashboard locally.

## 5) Common troubleshooting

### Torch / DLL import issues

If you see a Windows DLL error while importing PyTorch, create a fresh environment and install the CPU build there.

```powershell
conda create -n megathon python=3.11 -y
conda activate megathon
pip install ultralytics torch
```

### Dataset validation errors

If the script reports missing folders, ensure this structure exists:

```text
archive/person-3/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
  data.yaml
```

### Training stops early

Check the output in:

```text
runs/train/person_detection/
```

Look for the generated metrics, checkpoint, and training logs.

## 6) Typical workflow

```powershell
cd C:\Users\admin\Downloads\megathon
.\.venv\Scripts\Activate.ps1
python train_person_yolo.py
python live_rescue_web_app.py
```

## 7) Useful references

- Ultralytics YOLOv8: https://docs.ultralytics.com/
- PyTorch: https://pytorch.org/
  ## 8) RESULTS :
  <img width="1743" height="921" alt="image" src="https://github.com/user-attachments/assets/cdcecf06-0a5f-4b06-bba5-54667a683a78" />


## License

This project is provided for local research and rescue simulation use.

# Aqua Culture Fish Detection 🐟

This project uses YOLOv8 to detect and classify fish species for aquaculture monitoring.

## Files
- `app.py` – Streamlit web app for prediction
- `train.py` – YOLOv8 training script
- `data.yaml` – Dataset config
- `yolov8n.pt` – Trained model weights
- Folders: `train/`, `valid/`, `test/` – Contain dataset images and labels

## Usage
```bash
streamlit run app.py

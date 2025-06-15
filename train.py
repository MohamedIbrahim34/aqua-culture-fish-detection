from ultralytics import YOLO
import yaml
import subprocess
import time

# Paths
DATA_YAML_PATH = 'data.yaml'  # Your data.yaml file
MODEL_TYPE = 'yolov8n.pt'             # Try yolov8s.pt or yolov8m.pt for better accuracy
EPOCHS = 70
PROJECT_NAME = 'fish_training'
EXPERIMENT_NAME = 'fish_detector'

# Load and train the model
model = YOLO(MODEL_TYPE)

results = model.train(
    data=DATA_YAML_PATH,
    epochs=EPOCHS,
    imgsz=640,
    batch=16,
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    verbose=True
)

# Start TensorBoard to visualize training
logdir = f'{PROJECT_NAME}/{EXPERIMENT_NAME}'
print(f"\n🎯 Starting TensorBoard for logs in: {logdir}\n")
subprocess.Popen(["tensorboard", "--logdir", logdir])  
time.sleep(5)
print("🔗 Open this link in your browser: http://localhost:6006")

# Validate model and print precision
metrics = model.val()
print(f"\n✅ Final Precision on validation set: {metrics.box.precision:.4f}")
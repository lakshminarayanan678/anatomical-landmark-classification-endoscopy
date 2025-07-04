import wandb
from ultralytics import YOLO
import os

# Initialize W&B run
run = wandb.init(project="capsule_vision_challenge_2024", name="yolo11n_cls_run", job_type="train")

# # Log dataset as artifact
# artifact = wandb.Artifact(name="anat_1_yolo_split", type="dataset")
# artifact.add_dir("path/to/mnist160")
# run.log_artifact(artifact)
# artifact.wait()

# Use existing dataset artifact
artifact = run.use_artifact("anat_1_yolo_split:latest", type="dataset")
# dataset_dir = artifact.download()

# Train the model
model = YOLO("yolo11n-cls.pt")
results = model.train(data="/home/endodl/PHASE-1/mln/anatomical/data", epochs=50, imgsz=224)

# Log the trained model as an artifact
model_artifact = wandb.Artifact(name="yolo11n_cls_model", type="model")
model_artifact.add_file(model.ckpt_path if hasattr(model, "ckpt_path") else "runs/classify/train/weights/best.pt")
run.log_artifact(model_artifact)

run.finish()
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("/home/endodl/PHASE-1/mln/anatomical/anatomical_stomach/anat_yolo_split/runs/classify/train/weights/best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("/home/endodl/PHASE-1/mln/anatomical/anatomical_stomach/anat_yolo_split/data/test/zline/2_zline.jpg", save=True, imgsz=64, conf=0.5)
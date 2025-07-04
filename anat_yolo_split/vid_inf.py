from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/split_data_yolov11ncls/V11l/runs/classify/train3/weights/best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("/home/endodl/PHASE-1/mln/data/val/org_vid_anat.mp4", save=True, imgsz=320, conf=0.5, show= True)
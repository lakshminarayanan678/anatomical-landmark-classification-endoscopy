from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/home/endodl/PHASE-1/mln/anatomical/anatomical_stomach/anat_yolo_split/yolov11n.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'

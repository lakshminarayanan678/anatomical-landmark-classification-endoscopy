from ultralytics import YOLO
# model = YOLO('/home/endodl/PHASE-1/mln/anatomical/mln_codes_gastrohun/runs/classify/train2/weights/best.pt')
model = YOLO("/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/split_data_yolov11ncls/V11l/runs/classify/train3/weights/best.pt")
test_results = model.val(data="/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/split_data_yolov11ncls/data1", save_json=True, plots=True, split="test")
# anatomical-landmark-classification-endoscopy
Endoscopy anatomical landmark classification using SOTA models

This project provides a deep learning pipeline to classify anatomical landmarks from endoscopic video and image frames using state-of-the-art vision models such as ConvNeXt, DINO-ViT, and YOLO

## Folder Structure

anatomical_stomach/
├── anat_convnext/ # ConvNeXt-based classification
│ └── ConvNeXt/
├── anat_dinovit/ # DINO-ViT classification model
├── anat_yolo_split/ # YOLOvX models adapted for anatomical classification

## Models Used

| Model       | Type                       | Description |
|-------------|----------------------------|-------------|
| ConvNeXt   | Image Classification        | Fine-tuned on anatomical landmark datasets |
| DINO-ViT   | Self-Supervised Transformer | Robust to variation in views and textures |
| YOLOvX     | Object Detection            | Real-time classification and localization |

---

## How to Run

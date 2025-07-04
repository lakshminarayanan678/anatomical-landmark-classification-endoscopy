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

### 1. Clone the Repository
```bash
git clone https://github.com/lakshminarayanan678/anatomical-landmark-classification-endoscopy.git
cd anatomical-landmark-classification-endoscopy

2. Install Requirements
Each model folder contains its own requirements.txt:

bash
Copy
Edit
cd anat_convnext
pip install -r requirements.txt
Repeat similarly for other subfolders.

3. Inference & Training
For ConvNeXt:

bash
Copy
Edit
cd anat_convnext/ConvNeXt
python train_model.py      # Train the model
python val_scores.py       # Evaluate performance
For DINO-ViT:

bash
Copy
Edit
cd anat_dinovit
python train_model.py
For YOLO:

bash
Copy
Edit
cd anat_yolo_split
# Usage depends on the YOLO variant
📦 Dataset
The models are trained and tested on labeled endoscopic frames containing anatomical landmarks.

Supports frame-based datasets (image folders with CSV annotations) and video-to-frame pipelines.

📈 Metrics Logged
Accuracy

Precision, Recall, F1 Score

Confusion Matrix

mAP@50 (YOLO models)

🧪 Sample Output
You can find example predictions and visualizations inside each model folder under results/.

📄 License
MIT License. See LICENSE file for more details.

👨‍💻 Author
Lakshminarayanan M – Biomedical Deep Learning Researcher

📬 lakshminarayanan678

📌 To-Do
 Add pretrained model weights via GitHub Releases or LFS

 Integrate real-time video classification

 Upload full benchmark results across all models

yaml
Copy
Edit

---

### 🔧 How to Add

Save this content as `README.md` in your repo root.

Then run:
```bash
git add README.md
git commit -m "Added project README with structure and instructions"
git push origin main

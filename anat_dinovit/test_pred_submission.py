import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

# load dino model
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# Define class labels (based on folder names)
class_columns = ['duodenalbulb', 'esophagus', 'pylorus', 'stomach', 'zline']

# Define the model architecture
class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = dinov2_vits14  # Assuming dinov2_vits14 is already defined elsewhere
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # Adjust the number of classes as needed
        )
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

# Load the model
def load_model(model_path, device):
    model = DinoVisionTransformerClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    return model.to(device)

# Define the image transformations
def get_transforms():
    return transforms.Compose([
        transforms.Resize(280),
        transforms.CenterCrop(280),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Load image paths and their ground truth labels
def load_test_images_with_labels(root_folder):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(root_folder))
    for label in class_names:
        class_path = os.path.join(root_folder, label)
        for img_name in os.listdir(class_path):
            if img_name.endswith('.jpg'):
                image_paths.append(os.path.join(class_path, img_name))
                labels.append(label)
    return image_paths, labels

# # Load and process the test images
# def load_test_images(image_folder):
#     image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')]
#     return image_paths

# # Make predictions
# def make_predictions(model, image_paths, device):
#     class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 
#                      'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    
#     all_image_paths = []
#     all_predictions = []
    
#     data_transforms = get_transforms()

#     with torch.no_grad():
#         for img_path in image_paths:
#             image = Image.open(img_path).convert('RGB')
#             input_tensor = data_transforms(image).unsqueeze(0).to(device)
            
#             outputs = model(input_tensor)
#             probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
#             predicted_class = np.argmax(probabilities, axis=1)

#             all_image_paths.append(os.path.basename(img_path))
#             all_predictions.append(probabilities[0])

#     return all_image_paths, np.array(all_predictions), class_columns


# Predict classes and probabilities
def make_predictions(model, image_paths, device, class_columns):
    all_predictions = []
    predicted_classes = []

    data_transforms = get_transforms()
    
    total_time = 0.0


    with torch.no_grad():
        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')
            input_tensor = data_transforms(image).unsqueeze(0).to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()    
            start_time = time.time()
            outputs = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted_class = np.argmax(probabilities, axis=1)
            predicted_classes.append(class_columns[predicted_class[0]])
            all_predictions.append(probabilities[0])
            
    avg_time_per_image = (total_time / len(image_paths)) * 1000 
    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Average inference time per image: {avg_time_per_image:.2f} ms")

    return predicted_classes, np.array(all_predictions)

# Save predictions and probabilities to Excel
def save_predictions_to_excel(image_paths, predictions, pred_labels, class_columns, output_path):
    df_prob = pd.DataFrame(predictions, columns=class_columns)
    df_prob.insert(0, 'image_path', image_paths)
    df_prob['predicted_class'] = pred_labels
    df_prob.to_excel(output_path, index=False, engine='openpyxl')

# Plot and save both raw and normalized confusion matrices
def plot_and_save_confusion_matrices(true_labels, pred_labels, class_columns, output_dir):
    cm = confusion_matrix(true_labels, pred_labels, labels=class_columns)

    # Raw matrix
    disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_columns)
    disp_raw.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix (Raw Counts)")
    raw_img_path = f"{output_dir}/confusion_matrix_raw.png"
    plt.savefig(raw_img_path, dpi=300, bbox_inches='tight')
    plt.close()

    df_cm_raw = pd.DataFrame(cm, index=class_columns, columns=class_columns)
    df_cm_raw.to_csv(f"{output_dir}/confusion_matrix_raw.csv")

    # Normalized matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_columns)
    disp_norm.plot(cmap=plt.cm.Greens, xticks_rotation=45, values_format=".2f")
    plt.title("Confusion Matrix (Normalized)")
    norm_img_path = f"{output_dir}/confusion_matrix_normalized.png"
    plt.savefig(norm_img_path, dpi=300, bbox_inches='tight')
    plt.close()

    df_cm_norm = pd.DataFrame(cm_normalized, index=class_columns, columns=class_columns)
    df_cm_norm.to_csv(f"{output_dir}/confusion_matrix_normalized.csv")

    print(f"Raw confusion matrix saved to {raw_img_path} and .csv")
    print(f"Normalized confusion matrix saved to {norm_img_path} and .csv")

if __name__ == "__main__":
    model_path = "/home/endodl/PHASE-1/mln/anatomical/anatomical_stomach/anat_dinovit/results/Anatomical_75epochs_dino_vit.pth"
    test_image_root = '/home/endodl/PHASE-1/mln/anatomical/test'  # folder with subfolders as class names
    excel_output_path = "/home/endodl/PHASE-1/mln/anatomical/anatomical_stomach/anat_dinovit/results/Test_results.xlsx"
    cm_output_dir = "/home/endodl/PHASE-1/mln/anatomical/anatomical_stomach/anat_dinovit/results"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    image_paths, true_labels = load_test_images_with_labels(test_image_root)
    predicted_labels, all_probs = make_predictions(model, image_paths, device, class_columns)

    save_predictions_to_excel(image_paths, all_probs, predicted_labels, class_columns, excel_output_path)

    plot_and_save_confusion_matrices(true_labels, predicted_labels, class_columns, cm_output_dir)

    print("All done.")
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
import json
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn

# load dino model
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# Define the model architecture
class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = dinov2_vits14  # Assuming dinov2_vits14 is already defined elsewhere
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # Adjust the number of classes as needed
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

# Function to save predictions to Excel
def save_predictions_to_excel(image_paths, y_pred, output_path):
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("y_pred:", y_pred)
    print("y_pred_classes",y_pred_classes)
    predicted_class_names = [class_columns[i] for i in y_pred_classes]
    print("predicted_class_names", predicted_class_names)
    df_prob = pd.DataFrame(y_pred, columns=class_columns)
    df_prob.insert(0, 'image_path', image_paths)
    df_class = pd.DataFrame({'image_path': image_paths, 'predicted_class': predicted_class_names})
    df_merged = pd.merge(df_prob, df_class, on='image_path')
    df_merged.to_excel(output_path, index=False)

# Specificity and Sensitivity Calculations
def calculate_specificity(y_true_class, y_pred_class):
    tn = np.sum((y_true_class == 0) & (y_pred_class == 0))
    fp = np.sum((y_true_class == 0) & (y_pred_class == 1))
    specificity = tn / (tn + fp + 1e-10)  # Avoid division by zero
    return specificity

def calculate_sensitivity(y_true_class, y_pred_class):
    tp = np.sum((y_true_class == 1) & (y_pred_class == 1))
    fn = np.sum((y_true_class == 1) & (y_pred_class == 0))
    sensitivity = tp / (tp + fn + 1e-10)  # Avoid division by zero
    return sensitivity

# Metrics Report Generation
def generate_metrics_report(y_true, y_pred):
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    metrics_report = {}

    # If y_pred contains probabilities, convert to class labels using argmax
    if y_pred.ndim > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred  # Already class labels
    
    y_true_classes = y_true  # Assuming these are class labels

    # Generate classification report
    class_report = classification_report(y_true_classes, y_pred_classes, target_names=class_columns, output_dict=True, zero_division=0)

    # Initialize metric holders
    auc_roc_scores = {}
    average_precision_scores = {}
    sensitivity_scores = {}
    specificity_scores = {}
    f1_scores = {}

    # Calculate per-class metrics
    for i, class_name in enumerate(class_columns):
        y_true_class = (y_true_classes == i).astype(int)
        y_pred_class = (y_pred_classes == i).astype(int)

        # AUC-ROC
        if y_pred.ndim > 1:
            auc_roc_scores[class_name] = roc_auc_score(y_true_class, y_pred[:, i])
        else:
            auc_roc_scores[class_name] = roc_auc_score(y_true_class, y_pred_class)

        # Average Precision
        average_precision_scores[class_name] = average_precision_score(y_true_class, y_pred_class)

        # Specificity
        specificity_scores[class_name] = calculate_specificity(y_true_class, y_pred_class)

        # Sensitivity (Recall)
        sensitivity_scores[class_name] = calculate_sensitivity(y_true_class, y_pred_class)

        # F1 Score
        f1_scores[class_name] = f1_score(y_true_class, y_pred_class)

    # Mean values of each metric
    mean_auc_roc = np.mean(list(auc_roc_scores.values()))
    mean_specificity = np.mean(list(specificity_scores.values()))
    mean_average_precision = np.mean(list(average_precision_scores.values()))
    mean_sensitivity = np.mean(list(sensitivity_scores.values()))
    mean_f1_score = np.mean(list(f1_scores.values()))

    # Balanced accuracy
    balanced_accuracy_scores = balanced_accuracy_score(y_true_classes, y_pred_classes)

    # Update the metrics report
    metrics_report.update(class_report)
    metrics_report['auc_roc_scores'] = auc_roc_scores
    metrics_report['specificity_scores'] = specificity_scores
    metrics_report['average_precision_scores'] = average_precision_scores
    metrics_report['sensitivity_scores'] = sensitivity_scores
    metrics_report['f1_scores'] = f1_scores
    metrics_report['mean_auc'] = mean_auc_roc
    metrics_report['mean_specificity'] = mean_specificity
    metrics_report['mean_average_precision'] = mean_average_precision
    metrics_report['mean_sensitivity'] = mean_sensitivity
    metrics_report['mean_f1_score'] = mean_f1_score
    metrics_report['balanced_accuracy'] = balanced_accuracy_scores

    metrics_report_json = json.dumps(metrics_report, indent=4)
    return metrics_report_json

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize((280, 280)),  # Adjust size as needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization for pre-trained models
])

# Load dataset and create DataLoader
test_dataset = ImageFolder(root='Dataset-challnge/test', transform=data_transforms)  # Adjust path
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save_path = "models/dino_vit_classifier_capsule.pth"
model = load_model(model_save_path, device)

# Set to evaluation mode
model.eval()

# Prepare for test set evaluation
all_image_paths = []  # Store image paths
all_predictions = []  # Store predictions
all_labels = []  # Store true labels

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:  # Use test_loader defined above
        images, labels = data
        all_labels.append(labels.cpu().numpy())  # Store true labels
        
        outputs = model(images.to(device))
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.append(probabilities)  # Store probabilities
        
        total += labels.size(0)
        correct += (predicted.to("cpu") == labels).sum().item()

# Flatten lists of predictions and labels
all_predictions = np.concatenate(all_predictions, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Calculate Accuracy
accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy:.2f}%')

# Generate the metrics report
metrics_report = generate_metrics_report(all_labels, all_predictions)
print(metrics_report)

# Save predictions to Excel
output_excel_path = "val_predictions_output.xlsx"
image_paths = [test_dataset.imgs[i][0] for i in range(len(all_predictions))]  # Actual image paths
save_predictions_to_excel(image_paths, all_predictions, output_excel_path)
print(f"Predictions saved to {output_excel_path}")

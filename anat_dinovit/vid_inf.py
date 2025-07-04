import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os
from tqdm import tqdm
import time

# Load DINO model from torch hub
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# Define class labels (update accordingly)
class_columns = ['duodenalbulb', 'esophagus', 'pylorus', 'stomach', 'zline']

# Define the classifier model
class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = dinov2_vits14
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, len(class_columns))
        )
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

# Load the trained model
def load_model(model_path, device):
    model = DinoVisionTransformerClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

# Define image transforms for a frame
def get_transforms():
    return transforms.Compose([
        transforms.Resize(280),
        transforms.CenterCrop(280),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Convert OpenCV BGR frame to Tensor
def preprocess_frame(frame, transform, device):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

# # Inference on video frames
# def infer_on_video(video_path, model, device, output_path=None, draw_prob=False):
#     transform = get_transforms()
#     cap = cv2.VideoCapture(video_path)
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps    = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"width: {width}, height: {height}, fps: {fps}, total frames: {total_frames}")

#     if output_path:
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#     else:
#         out = None

#     font = cv2.FONT_HERSHEY_SIMPLEX

#     with torch.no_grad():
#         frame_count = 0
#         preprocess_times = []
#         inference_times = []
#         draw_times = []
#         write_times = []

#         total_start_time = time.time()

#         for _ in tqdm(range(total_frames), desc="Processing video"):
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Preprocess
#             t0 = time.time()
#             input_tensor = preprocess_frame(frame, transform, device)
#             tx = time.time()
#             preprocess_times.append(tx - t0)

#             # Inference
#             if device.type == 'cuda':
#                 torch.cuda.synchronize()
#             t1 = time.time()
#             outputs = model(input_tensor)
#             if device.type == 'cuda':
#                 torch.cuda.synchronize()
#             ty= time.time()
#             inference_times.append(ty - t1)

#             # Prediction
#             probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
#             pred_class = class_columns[np.argmax(probs)]

#             # Annotate
#             t2 = time.time()
#             text = f"{pred_class} ({np.max(probs)*100:.1f}%)"
#             cv2.putText(frame, text, (30, 50), font, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
#             if draw_prob:
#                 for i, cls in enumerate(class_columns):
#                     prob_text = f"{cls}: {probs[i]*100:.1f}%"
#                     cv2.putText(frame, prob_text, (30, 80 + i*30), font, 0.6, (255, 255, 255), 1)
#             draw_times.append(time.time() - t2)

#             # Write
#             t3 = time.time()
#             if out:
#                 out.write(frame)
#             write_times.append(time.time() - t3)

#             frame_count += 1

#         total_end_time = time.time()
#         total_duration = total_end_time - total_start_time
#         model_duration = ty - total_start_time
#         fps_whole = frame_count / total_duration
#         fps_model = frame_count / model_duration

#         print(f"\n==== Inference Summary ====")
#         print(f"Total frames processed: {frame_count}")
#         print(f"Total duration: {total_duration:.2f} sec")
#         print(f"Model duration: {model_duration:.2f} sec")
#         print(f"Overall FPS: {fps_whole:.2f}")
#         print(f"Model FPS: {fps_model:.2f}")
#         print(f"Average times per frame:")
#         print(f"  Preprocess: {np.mean(preprocess_times)*1000:.2f} ms")
#         print(f"  Inference : {np.mean(inference_times)*1000:.2f} ms")
#         print(f"  Annotate  : {np.mean(draw_times)*1000:.2f} ms")
#         print(f"  Write     : {np.mean(write_times)*1000:.2f} ms")

#     cap.release()
#     if out:
#         out.release()
#         print(f"Annotated video saved to {output_path}")
#     cv2.destroyAllWindows()

def infer_on_video_live(video_path, model, device, draw_prob=True):
    transform = get_transforms()
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"width: {width}, height: {height}, fps: {fps}, total frames: {total_frames}")

    font = cv2.FONT_HERSHEY_SIMPLEX

    with torch.no_grad():
        frame_count = 0
        preprocess_times = []
        inference_times = []
        draw_times = []

        total_start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # === Preprocess ===
            t0 = time.time()
            input_tensor = preprocess_frame(frame, transform, device)
            t1 = time.time()
            preprocess_times.append(t1 - t0)

            # === Inference ===
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t2 = time.time()

            outputs = model(input_tensor)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t3 = time.time()
            inference_times.append(t3 - t2)

            # === Prediction ===
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = class_columns[np.argmax(probs)]

            # === Annotate ===
            t4 = time.time()
            text = f"{pred_class} ({np.max(probs)*100:.1f}%)"
            cv2.putText(frame, text, (30, 50), font, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
            if draw_prob:
                for i, cls in enumerate(class_columns):
                    prob_text = f"{cls}: {probs[i]*100:.1f}%"
                    cv2.putText(frame, prob_text, (30, 80 + i*30), font, 0.6, (255, 255, 255), 1)
            draw_times.append(time.time() - t4)

            # === Display ===
            cv2.imshow("Live Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        total_model_time = sum(preprocess_times) + sum(inference_times)

        fps_whole = frame_count / total_duration
        fps_model = frame_count / total_model_time

        print(f"\n==== Inference Summary ====")
        print(f"Total frames processed: {frame_count}")
        print(f"Total duration: {total_duration:.2f} sec")
        print(f"Model duration (prep + inference): {total_model_time:.2f} sec")
        print(f"Overall FPS: {fps_whole:.2f}")
        print(f"Model FPS: {fps_model:.2f}")
        print(f"Average times per frame:")
        print(f"  Preprocess: {np.mean(preprocess_times)*1000:.2f} ms")
        print(f"  Inference : {np.mean(inference_times)*1000:.2f} ms")
        print(f"  Annotate  : {np.mean(draw_times)*1000:.2f} ms")

    cap.release()
    cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    model_path = "/home/endodl/PHASE-1/mln/anatomical/anatomical_stomach/anat_dinovit/results/Anatomical_75epochs_dino_vit.pth"
    video_path = "/home/endodl/PHASE-1/mln/data/val/org_vid_anat.mp4"
    # output_path = "/home/endodl/PHASE-1/mln/anatomical/anatomical_stomach/anat_dinovit/video_inf/video_inf1.avi"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    infer_on_video_live(video_path, model, device, draw_prob=True)
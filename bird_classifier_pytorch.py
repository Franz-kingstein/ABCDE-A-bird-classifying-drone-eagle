"""
PyTorch Bird Classifier Inference Script
Uses the converted .pt model for real-time bird detection and classification
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import time
import json
from pathlib import Path

# ============= MODEL DEFINITION =============
class EfficientNetV2B2PyTorch(nn.Module):
    def __init__(self, num_classes=25):
        super(EfficientNetV2B2PyTorch, self).__init__()
        
        # Load pretrained EfficientNetV2B2 from torchvision
        from torchvision.models import efficientnet_v2_b2, EfficientNet_V2_B2_Weights
        base_model = efficientnet_v2_b2(weights=EfficientNet_V2_B2_Weights.IMAGENET1K_V1)
        
        # Remove the classification head
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        # Custom classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1408, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


# ============= UTILITIES =============
def load_model(model_path='bird_classifier.pt', device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load PyTorch model from .pt file"""
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device)
    model = EfficientNetV2B2PyTorch(num_classes=25)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    class_names = checkpoint.get('class_names', [])
    print(f"✅ Model loaded successfully on {device}!")
    
    return model, class_names


def preprocess_image(image, target_size=(300, 300)):
    """Preprocess image for inference"""
    # Resize
    resized = cv2.resize(image, target_size)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize (ImageNet normalization)
    normalized = rgb.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    
    return tensor


def predict(model, image, class_names, device='cpu', suppress_class=None):
    """Run inference on image"""
    # Preprocess
    tensor = preprocess_image(image)
    tensor = tensor.to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
    
    probs = probabilities.cpu().numpy()[0]
    
    # Suppress a specific class if needed (e.g., for debugging)
    if suppress_class is not None and suppress_class in class_names:
        idx = class_names.index(suppress_class)
        probs[idx] = 0
        probs = probs / probs.sum()  # Renormalize
    
    # Get top predictions
    top_idx = np.argmax(probs)
    top_confidence = probs[top_idx]
    top_class = class_names[top_idx]
    
    return top_class, top_confidence, probs


# ============= MAIN INFERENCE LOOP =============
def run_inference(model_path='bird_classifier.pt', webcam_index=0, suppress_class=None):
    """Main inference loop for webcam"""
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model, class_names = load_model(model_path, device)
    
    # Initialize webcam
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {webcam_index}")
        return
    
    print("Starting inference. Press 'q' to quit.")
    
    bird_counts = {}
    detection_log = []
    active_tracks = {}
    next_track_id = 0
    verification_threshold = 5  # seconds
    track_expiry = 2.0  # seconds
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Simple contour detection for ROI
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            found_this_frame = []
            now = time.time()
            
            # Process each contour
            for cnt in contours:
                x, y, roi_w, roi_h = cv2.boundingRect(cnt)
                if roi_w < 60 or roi_h < 60:
                    continue
                
                roi = frame[y:y+roi_h, x:x+roi_w]
                if roi.size == 0:
                    continue
                
                # Run inference on ROI
                bird_class, confidence, _ = predict(model, roi, class_names, device, suppress_class)
                
                centroid = (x + roi_w//2, y + roi_h//2)
                found_this_frame.append({
                    "name": bird_class,
                    "conf": confidence,
                    "box": (x, y, roi_w, roi_h),
                    "centroid": centroid
                })
            
            # Simple tracking with centroid matching
            for bird in found_this_frame:
                best_match_id = None
                min_dist = 100
                
                for tid, track in active_tracks.items():
                    dist = np.sqrt((bird["centroid"][0] - track["centroid"][0])**2 + 
                                  (bird["centroid"][1] - track["centroid"][1])**2)
                    if dist < min_dist and bird["name"] == track["name"]:
                        min_dist = dist
                        best_match_id = tid
                
                if best_match_id is not None:
                    track = active_tracks[best_match_id]
                    track.update({"centroid": bird["centroid"], "last_seen": now, "box": bird["box"]})
                    
                    elapsed = now - track["start_time"]
                    if elapsed >= verification_threshold and not track["is_verified"]:
                        bird_counts[bird["name"]] = bird_counts.get(bird["name"], 0) + 1
                        detection_log.append({
                            "time": time.strftime("%H:%M:%S"),
                            "name": bird["name"],
                            "count": bird_counts[bird["name"]]
                        })
                        track["is_verified"] = True
                        print(f"✅ VERIFIED: {bird['name']} at {detection_log[-1]['time']}")
                else:
                    active_tracks[next_track_id] = {
                        "name": bird["name"],
                        "start_time": now,
                        "is_verified": False,
                        "last_seen": now,
                        "centroid": bird["centroid"],
                        "box": bird["box"]
                    }
                    next_track_id += 1
            
            # Cleanup expired tracks
            active_tracks = {tid: t for tid, t in active_tracks.items() 
                            if now - t["last_seen"] < track_expiry}
            
            # Draw boxes and labels
            for tid, track in active_tracks.items():
                x, y, roi_w, roi_h = track["box"]
                color = (0, 255, 0) if track["is_verified"] else (0, 255, 255)  # Green or Cyan
                label = f"{track['name']} ({int(now - track['start_time'])}s)"
                
                cv2.rectangle(display_frame, (x, y), (x+roi_w, y+roi_h), color, 2)
                cv2.putText(display_frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw tally
            y_offset = 30
            cv2.putText(display_frame, "VERIFIED TALLY:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for name, count in bird_counts.items():
                y_offset += 25
                cv2.putText(display_frame, f"{name}: {count}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("Bird Classifier (PyTorch)", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Inference complete!")


if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'bird_classifier.pt'
    webcam_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    suppress_class = sys.argv[3] if len(sys.argv) > 3 else None
    
    print(f"PyTorch Bird Classifier")
    print(f"Model: {model_path}")
    print(f"Webcam: {webcam_index}")
    if suppress_class:
        print(f"Suppressing class: {suppress_class}")
    print("-" * 50)
    
    run_inference(model_path, webcam_index, suppress_class)

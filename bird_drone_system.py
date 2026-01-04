import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B2, MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout
import numpy as np
import cv2
import time
import h5py
import os

# --- CONFIGURATION ---
CLASS_NAMES = [
    "Asian-Green-Bee-Eater", "Brown-Headed-Barbet", "Cattle-Egret", "Common-Kingfisher",
    "Common-Myna", "Common-Rosefinch", "Common-Tailorbird", "Coppersmith-Barbet",
    "Forest-Wagtail", "Gray-Wagtail", "Hoopoe", "House-Crow", "Indian-Grey-Hornbill",
    "Indian-Peacock", "Indian-Pitta", "Indian-Roller", "Jungle-Babbler", "Northern-Lapwing",
    "Red-Wattled-Lapwing", "Ruddy-Shelduck", "Rufous-Treepie", "Sarus-Crane",
    "White-Breasted-Kingfisher", "White-Breasted-Waterhen", "White-Wagtail"
]

BIRD_KEYWORDS = [
    'bird', 'finch', 'bulbul', 'jay', 'magpie', 'chickadee', 'kite', 'eagle', 
    'vulture', 'owl', 'grouse', 'peacock', 'quail', 'parrot', 'toucan', 
    'duck', 'goose', 'swan', 'stork', 'heron', 'egret', 'crane', 'pelican', 
    'penguin', 'albatross', 'kingfisher', 'myna', 'bee eater', 'hornbill'
]

# --- MODEL LOADING ---
def build_and_load_model(weights_path='weights.h5'):
    print("Building EfficientNetV2B2 model...")
    base_model = EfficientNetV2B2(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(), 
        Dense(256),  
        BatchNormalization(),  
        Activation('relu'),  
        Dropout(0.3),  
        Dense(128), 
        BatchNormalization(),
        Activation('relu'),  
        Dropout(0.3),
        Dense(25, activation='softmax')  
    ])

    print(f"Loading weights from {weights_path}...")
    try:
        with h5py.File(weights_path, 'r') as f:
            # Manual mapping to model layers based on weights.h5 structure
            model.layers[2].set_weights([f['layers']['dense']['vars']['0'][:], f['layers']['dense']['vars']['1'][:]])
            model.layers[3].set_weights([f['layers']['batch_normalization']['vars']['0'][:], f['layers']['batch_normalization']['vars']['1'][:], f['layers']['batch_normalization']['vars']['2'][:], f['layers']['batch_normalization']['vars']['3'][:]])
            model.layers[6].set_weights([f['layers']['dense_1']['vars']['0'][:], f['layers']['dense_1']['vars']['1'][:]])
            model.layers[7].set_weights([f['layers']['batch_normalization_1']['vars']['0'][:], f['layers']['batch_normalization_1']['vars']['1'][:], f['layers']['batch_normalization_1']['vars']['2'][:], f['layers']['batch_normalization_1']['vars']['3'][:]])
            model.layers[10].set_weights([f['layers']['dense_2']['vars']['0'][:], f['layers']['dense_2']['vars']['1'][:]])
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None
    return model

# --- UTILITIES ---
def get_real_world_location():
    # Placeholder for real GPS data
    return "Lat: 12.9716, Long: 77.5946 (Mock Location)"

# --- MAIN INFERENCE LOOP ---
def run_inference():
    model = build_and_load_model()
    if model is None: return

    print("Loading bird detector (MobileNetV2)...")
    detector = MobileNetV2(weights='imagenet')

    cap = cv2.VideoCapture(0) # Try 0 first for script
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

    bird_counts = {}
    detection_log = []
    active_tracks = {}
    next_track_id = 0
    verification_threshold = 5
    track_expiry = 2.0

    print("Starting webcam. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simple contour detection for ROIs
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            found_this_frame = []

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 60 or h < 60: continue # Filter noise
                
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0: continue
                
                # Stage 1: Is it a bird?
                detect_roi = cv2.resize(roi, (224, 224))
                detect_input = preprocess_input(np.array(detect_roi, dtype=np.float32))
                detect_input = np.expand_dims(detect_input, axis=0)
                
                detect_preds = detector.predict(detect_input, verbose=0)
                decoded = decode_predictions(detect_preds, top=3)[0]
                
                is_bird = any(any(key in label.lower().replace('_', ' ') for key in BIRD_KEYWORDS) for _, label, score in decoded)

                if is_bird:
                    # Stage 2: Which bird?
                    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    resized_roi = cv2.resize(rgb_roi, (300, 300))
                    input_array = np.array(resized_roi, dtype=np.float32)
                    input_array = np.expand_dims(input_array, axis=0)

                    preds = model.predict(input_array, verbose=0)
                    
                    # Suppress 'Northern-Lapwing' (Index 17) for debugging
                    lapwing_idx = CLASS_NAMES.index("Northern-Lapwing")
                    preds[0][lapwing_idx] = 0
                    
                    class_idx = np.argmax(preds)
                    confidence = np.max(preds)
                    name = CLASS_NAMES[class_idx]
                    centroid = (x + w//2, y + h//2)
                    found_this_frame.append({"name": name, "conf": confidence, "box": (x, y, w, h), "centroid": centroid})

            # Tracking Logic
            now = time.time()
            for bird in found_this_frame:
                best_match_id = None
                min_dist = 100
                
                for tid, track in active_tracks.items():
                    dist = np.sqrt((bird["centroid"][0] - track["centroid"][0])**2 + (bird["centroid"][1] - track["centroid"][1])**2)
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
                            "count": bird_counts[bird["name"]],
                            "location": get_real_world_location()
                        })
                        track["is_verified"] = True
                        print(f"VERIFIED: {bird['name']} at {detection_log[-1]['time']}")
                else:
                    active_tracks[next_track_id] = {
                        "name": bird["name"], "start_time": now, "is_verified": False,
                        "last_seen": now, "centroid": bird["centroid"], "box": bird["box"]
                    }
                    next_track_id += 1

            # Cleanup and UI
            active_tracks = {tid: t for tid, t in active_tracks.items() if now - t["last_seen"] < track_expiry}

            for tid, track in active_tracks.items():
                x, y, w, h = track["box"]
                color = (0, 255, 0) if track["is_verified"] else (0, 255, 255)
                label = f"{track['name']} ({int(now - track['start_time'])}s)"
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Overlay Tally
            y_off = 30
            cv2.putText(display_frame, "TALLY:", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for name, count in bird_counts.items():
                y_off += 25
                cv2.putText(display_frame, f"{name}: {count}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Bird Detection Drone System", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()

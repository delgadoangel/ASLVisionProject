import cv2
import mediapipe as mp
import numpy as np
import joblib
import random
import time
import os
from tensorflow.keras.models import load_model

# Load CNN model and label encoder
model = load_model("cnn_gesture_model.keras")
le = joblib.load("cnn_label_encoder.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Game state
target_letter = random.choice(le.classes_)
score = 0
show_hint = False

# Initialize camera
cap = cv2.VideoCapture(0)

def create_celeb_mask(celeb_frame):
    #get channels
    b = celeb_frame[:, :, 0]
    g = celeb_frame[:, :, 1]
    r = celeb_frame[:, :, 2]

    #create a mask where green is much higher than red and blue
    green_dominant = (g > r + 40) & (g > b + 40)

    #convert to 8-bit image (0-255)
    green_screen_mask = green_dominant.astype(np.uint8) * 255

    #invert the mask so UFO is white (255) and green background is black (0)
    celeb_mask = cv2.bitwise_not(green_screen_mask)

    # print(celeb_mask.shape)

    #mask has 3 channels
    celeb_mask = np.stack([celeb_mask, celeb_mask, celeb_mask], axis=2)

    #print(celeb_mask.shape)
    return celeb_mask

def overlay_celeb(background_frame, celeb_frame, celeb_mask):
    # Resize celeb frame and mask to match background
    celeb_frame = cv2.resize(celeb_frame, (background_frame.shape[1], background_frame.shape[0]))
    celeb_mask = cv2.resize(celeb_mask, (background_frame.shape[1], background_frame.shape[0]))

    # Invert mask
    inv_mask = cv2.bitwise_not(celeb_mask)

    # Extract foreground (celeb) and background
    fg = cv2.bitwise_and(celeb_frame, celeb_mask)
    bg = cv2.bitwise_and(background_frame, inv_mask)

    # Combine
    combined = cv2.add(bg, fg)
    return combined

def load_hint_image(letter):
    filename = f"hints/{letter.upper()}.jpeg"
    if os.path.exists(filename):
        hint = cv2.imread(filename)
        return cv2.resize(hint, (150, 150))
    return None


# Finger setup
fingertips = [4, 8, 12, 16, 20]
finger_segments = [
    [1, 2, 3, 4],    # Thumb
    [5, 6, 7, 8],    # Index
    [9,10,11,12],    # Middle
    [13,14,15,16],   # Ring
    [17,18,19,20]    # Pinky
]

# Distance function
def euclidean(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def compute_features(landmarks):
    features = []

    # 5 wrist-to-fingertip distances
    for i in [4, 8, 12, 16, 20]:
        features.append(euclidean(landmarks[0], landmarks[i]))

    # Thumb: 3 segments (1→2, 2→3, 3→4)
    features.append(euclidean(landmarks[1], landmarks[2]))
    features.append(euclidean(landmarks[2], landmarks[3]))
    features.append(euclidean(landmarks[3], landmarks[4]))

    # Index: 3 segments (5→6, 6→7, 7→8)
    features.append(euclidean(landmarks[5], landmarks[6]))
    features.append(euclidean(landmarks[6], landmarks[7]))
    features.append(euclidean(landmarks[7], landmarks[8]))

    # Middle: 3 segments (9→10, 10→11, 11→12)
    features.append(euclidean(landmarks[9], landmarks[10]))
    features.append(euclidean(landmarks[10], landmarks[11]))
    features.append(euclidean(landmarks[11], landmarks[12]))

    # Ring: 3 segments (13→14, 14→15, 15→16)
    features.append(euclidean(landmarks[13], landmarks[14]))
    features.append(euclidean(landmarks[14], landmarks[15]))
    features.append(euclidean(landmarks[15], landmarks[16]))

    # Pinky: 3 segments (17→18, 18→19, 19→20)
    features.append(euclidean(landmarks[17], landmarks[18]))
    features.append(euclidean(landmarks[18], landmarks[19]))
    features.append(euclidean(landmarks[19], landmarks[20]))

    return np.array(features).reshape(1, 20, 1)


def get_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        return compute_features(result.multi_hand_landmarks[0].landmark)
    return None

celeb_cap = cv2.VideoCapture("celeb.mp4")
#all frames from celeb video and store them in a list
celeb_frames = []
count = 0
while True:
    ret, frame = celeb_cap.read()
    count += 1
    if not ret or count > 300:
        break
    celeb_frames.append(frame)

#create celeb masks for each frame
celeb_masks = []
for frame in celeb_frames:
    celeb_masks.append(create_celeb_mask(frame))


show_celebration = False
celeb_index = 0
seen_letters = set()
hint_img = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Show celeb overlay if in celebration window
    if show_celebration:
        if celeb_index < len(celeb_frames):
            celeb_overlay = overlay_celeb(frame, celeb_frames[celeb_index], celeb_masks[celeb_index])
            celeb_index += 1
            frame = celeb_overlay
        else:
            show_celebration = False


    h, w, _ = frame.shape
    landmarks = get_landmarks(frame)

    predicted_letter = ""
    if landmarks is not None:
        pred = model.predict(landmarks, verbose=0)

        # Get all class probabilities
        class_probs = pred[0]

        # Create a mask to zero out probabilities of seen letters
        for i, letter in enumerate(le.classes_):
            if letter in seen_letters:
                class_probs[i] = 0

        # Get the highest probability among remaining letters
        if np.max(class_probs) > 0:  # Only if there are unseen letters left
            predicted_letter = le.inverse_transform([np.argmax(class_probs)])[0]
            confidence = np.max(class_probs)
            print(predicted_letter, confidence)

        # Check against target
        if predicted_letter == target_letter:
            correct_time = time.time()
            show_celebration = True
            celeb_index = 0

            # Update game state now
            score += 1
            seen_letters.add(target_letter)

            # Check if game is complete
            if len(seen_letters) == len(le.classes_):
                target_letter = None
                print("All letters complete")
            else:
                remaining = list(set(le.classes_) - seen_letters)
                target_letter = random.choice(remaining)

            show_hint = False
            hint_img = None


    # --- UI Updates ---

    # Top middle: letter prompt
    cv2.putText(frame, f"Show the letter: {target_letter}", (w//2 - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Below prompt: hint text
    cv2.putText(frame, "Press H to get a hint", (w//2 - 130, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Top right: score
    cv2.putText(frame, f"Score: {score}", (w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Bottom right: predicted letter
    if predicted_letter:
        cv2.putText(frame, f"Prediction: {predicted_letter}", (w - 250, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Display hint image if requested (optional: not implemented yet)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('h'):
        if show_hint:
            show_hint = False
            hint_img = None
        elif target_letter:
            hint_img = load_hint_image(target_letter)
            show_hint = True
    elif key == ord('s'):
        if target_letter:
            remaining = list(set(le.classes_) - seen_letters)
            if remaining:
                target_letter = random.choice(remaining)
                show_hint = False
                hint_img = None
    elif key == ord('q'):
        break


    # Show hint overlay if requested
    if show_hint and hint_img is not None:
        hint_h, hint_w = hint_img.shape[:2]
        x, y = 10, h - hint_h - 10
        frame[y:y+hint_h, x:x+hint_w] = hint_img

    cv2.imshow("ASL Game", frame)

cap.release()
cv2.destroyAllWindows()
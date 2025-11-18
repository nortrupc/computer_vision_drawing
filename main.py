import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = 0, 0
drawing_mode = False 
fingers_touching = False

button_x1, button_y1 = 10, 10   
button_x2, button_y2 = 150, 60  
button_color = (0, 255, 0)

button_x1_2, button_y1_2 = 500, 10   
button_x2_2, button_y2_2 = 640, 60  
button_color_2 = (0, 255, 0)

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def is_inside_button(x, y, x1, y1, x2, y2):
    return x1 < x < x2 and y1 < y < y2

def fingers_are_open(landmarks):
    index_tip = landmarks[8]
    index_knuckle = landmarks[6]
    thumb_tip = landmarks[4]
    thumb_knuckle = landmarks[2]
    return index_tip.y < index_knuckle.y and thumb_tip.y < thumb_knuckle.y

def fingers_are_touching(landmarks):
    index_tip = landmarks[8]
    thumb_tip = landmarks[4]
    return index_tip.y < thumb_tip.y 



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    cv2.rectangle(frame, (button_x1, button_y1), (button_x2, button_y2), button_color, -1)
    text = "Clear"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, font_thickness = 1, 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = button_x1 + (button_x2 - button_x1 - text_size[0]) // 2
    text_y = button_y1 + (button_y2 - button_y1 + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    cv2.rectangle(frame, (button_x1_2, button_y1_2), (button_x2_2, button_y2_2), button_color_2, -1)
    text_2 = "Detect"
    text_x_2 = button_x1_2 + (button_x2_2 - button_x1_2 - text_size[0]) // 2
    text_y_2 = button_y1_2 + (button_y2_2 - button_y1_2 + text_size[1]) // 2
    cv2.putText(frame, text_2, (text_x_2, text_y_2), font, font_scale, (255, 255, 255), font_thickness)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            h, w, _ = frame.shape
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            distance = calculate_distance(index_x, index_y, thumb_x, thumb_y)

            threshold = 40

            if distance < threshold:
                if not fingers_touching and fingers_are_touching(hand_landmarks.landmark):
                    drawing_mode = not drawing_mode  
                    print(f"Drawing mode {'ON' if drawing_mode else 'OFF'}")
                    if drawing_mode:
                        time.sleep(0.2) 
                fingers_touching = True
            else:
                drawing_mode = False
                fingers_touching = False
            if drawing_mode:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = index_x, index_y
                
                cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (0, 0, 255), 5)
                prev_x, prev_y = index_x, index_y
            else:
                prev_x, prev_y = 0, 0

            if is_inside_button(index_x, index_y, button_x1, button_y1, button_x2, button_y2):
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                print("Canvas cleared!")
                time.sleep(.1)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        combined_frame = cv2.add(frame, canvas)

        if is_inside_button(index_x, index_y, button_x1_2, button_y1_2, button_x2_2, button_y2_2) and not drawing_mode:
            print("Detecting...")
            success = cv2.imwrite("detected_image.png", canvas)
            if success:
                print("Image saved successfully!")
            else:
                print("Failed to save image.")
            time.sleep(.1)

        cv2.imshow("Finger Drawing", combined_frame)


    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

    
import cv2
import mediapipe as mp
import numpy as np

# 1. INITIALIZE
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
px, py = 0, 0 
draw_color = (255, 0, 255) # Default Purple
brush_thickness = 10

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Get landmarks for Index (8) and Middle (12) tips and their knuckles (6, 10)
            lm = hand_lms.landmark
            cx, cy = int(lm[8].x * w), int(lm[8].y * h)   # Index Tip
            mx, my = int(lm[12].x * w), int(lm[12].y * h) # Middle Tip

            # Check which fingers are up
            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y

            # 2. SELECTION MODE (Two fingers up)
            if index_up and middle_up:
                px, py = 0, 0 # Reset previous points so it doesn't "jump"
                cv2.rectangle(frame, (cx, cy - 25), (mx, my + 25), draw_color, cv2.FILLED)
                cv2.putText(frame, "Selection Mode", (cx, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 3. DRAWING MODE (Only Index finger up)
            elif index_up:
                cv2.circle(frame, (cx, cy), 10, draw_color, cv2.FILLED)
                if px == 0 and py == 0:
                    px, py = cx, cy
                
                cv2.line(canvas, (px, py), (cx, cy), draw_color, brush_thickness)
                px, py = cx, cy
            
            else:
                px, py = 0, 0

    # Merge Canvas with Video
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Virtual Painter", frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()

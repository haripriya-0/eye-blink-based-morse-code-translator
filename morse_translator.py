import cv2
import dlib
import joblib
import numpy as np
import time
from imutils import face_utils

# Morse dictionary
MORSE_CODE_DICT = {
    '.-':'A', '-...':'B', '-.-.':'C', '-..':'D', '.':'E',
    '..-.':'F', '--.':'G', '....':'H', '..':'I', '.---':'J',
    '-.-':'K', '.-..':'L', '--':'M', '-.':'N', '---':'O',
    '.--.':'P', '--.-':'Q', '.-.':'R', '...':'S', '-':'T',
    '..-':'U', '...-':'V', '.--':'W', '-..-':'X', '-.--':'Y',
    '--..':'Z', '-----':'0', '.----':'1', '..---':'2', '...--':'3',
    '....-':'4', '.....':'5', '-....':'6', '--...':'7', '---..':'8', '----.':'9'
}

# Load model
model = joblib.load('blink_classifier_model.pkl')

# dlib setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))
IMG_SIZE = 64

# Timers
blink_start_time = None
last_blink_end_time = time.time()
morse_sequence = ""
final_text = ""

def get_eye_prediction(gray, eye_points):
    (x, y, w, h) = cv2.boundingRect(np.array(eye_points))
    eye = gray[y:y+h, x:x+w]
    eye = cv2.resize(eye, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1, -1)
    prediction = model.predict(eye)[0]
    return prediction

# Webcam
cap = cv2.VideoCapture(0)
blink_active = False

print("Start blinking to enter Morse Code...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape_np = face_utils.shape_to_np(shape)

        left_eye = shape_np[LEFT_EYE_POINTS]
        right_eye = shape_np[RIGHT_EYE_POINTS]

        left_pred = get_eye_prediction(gray, left_eye)
        right_pred = get_eye_prediction(gray, right_eye)

        both_closed = (left_pred == 0 and right_pred == 0)

        current_time = time.time()

        if both_closed:
            if not blink_active:
                blink_start_time = current_time
                blink_active = True
        else:
            if blink_active:
                blink_duration = current_time - blink_start_time
                if blink_duration < 0.3:
                    morse_sequence += '.'
                    print("Detected DOT (.)")
                else:
                    morse_sequence += '-'
                    print("Detected DASH (-)")
                last_blink_end_time = current_time
                blink_active = False

        # Check for end of letter (gap > 1.0 sec)
        if not blink_active and (current_time - last_blink_end_time) > 2.0:
            if morse_sequence:
                letter = MORSE_CODE_DICT.get(morse_sequence, '?')
                final_text += letter
                print(f"Morse: {morse_sequence} -> Letter: {letter}")
                morse_sequence = ""

        # Check for end of word (gap > 4.0 sec)
        if not blink_active and (current_time - last_blink_end_time) > 4.0 and final_text and final_text[-1] != ' ':
            final_text += ' '
            print("Word gap detected.")

        # Show feedback
        cv2.putText(frame, f"Morse: {morse_sequence}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Text: {final_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Draw eye landmarks
        cv2.polylines(frame, [left_eye], True, (0,255,255), 1)
        cv2.polylines(frame, [right_eye], True, (0,255,255), 1)

    cv2.imshow("Blink to Morse Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Final Output: {final_text.strip()}")

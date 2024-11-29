import cv2
import numpy as np

# Ładowanie klasyfikatorów Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Uruchomienie kamerki
cap = cv2.VideoCapture(0)

# Zmienna do śledzenia ostatniej pozycji źrenicy
last_pupil_position = None

while True:
    # Odczyt obrazu z kamerki
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrywanie twarzy
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Rysowanie prostokąta wokół twarzy
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #cv2.rectangle(frame, (x, y), (x+w, y+h*2//3), (0, 0, 255), 2)
        #roi_gray = gray[y:y + h*2//3, x:x+w]
        #roi_color = frame[y:y + h*2//3, x:x+w]


        # Region twarzy do analizy oczu
        roi_gray = gray[y:y + h * 2 // 3, x:x + w]
        roi_color = frame[y:y + h * 2 // 3, x:x + w]

        # Wykrywanie oczu
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Rysowanie prostokąta wokół oczu
            #cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Progowanie
            _, threshold = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Znajdź największy kontur (źrenica)
                largest_contour = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

                # Tylko jeśli promień jest odpowiedni
                if radius > 3:
                    last_pupil_position = (int(ex + cx), int(ey + cy))
                    cv2.circle(eye_color, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)

            # Jeśli źrenica nie została wykryta, przewidź na podstawie ostatniej pozycji
            if last_pupil_position is not None and not contours:
                # Rysowanie predykcji na podstawie ostatniej znanej pozycji
                predicted_x = int(last_pupil_position[0] - ex)
                predicted_y = int(last_pupil_position[1] - ey)

                if 0 <= predicted_x < ew and 0 <= predicted_y < eh:
                    cv2.circle(eye_color, (predicted_x, predicted_y), 5, (255, 0, 0), 2)

    # Wyświetlanie obrazu
    cv2.imshow('Eye Tracking', frame)

    # Zatrzymanie programu po wciśnięciu 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

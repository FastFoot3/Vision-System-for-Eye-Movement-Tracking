import cv2
import numpy as np

# Ładowanie klasyfikatorów Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Parametry dla Optical Flow (Lucas-Kanade)
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Uruchomienie kamerki
cap = cv2.VideoCapture(0)

# Zmienna do przechowywania punktów źrenic (lista pozycji dla obu oczu)
pupil_positions = [None, None]
old_gray = None

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if old_gray is None:
        old_gray = gray.copy()

    # Wykrywanie twarzy
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Rysowanie prostokąta wokół twarzy
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h*2//3), (0, 0, 255), 2)


        # Region twarzy do analizy oczu

        roi_gray = gray[y:y + h * 2 // 3, x:x + w]
        roi_color = frame[y:y + h * 2 // 3, x:x + w]

        # Wykrywanie oczu
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=10)
        for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # Ograniczamy do dwóch oczu
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Rysowanie prostokąta wokół oczu
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Jeśli nie mamy pozycji źrenicy dla oka, wykryj ją
            if pupil_positions[i] is None:
                _, threshold = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                    if radius > 3:
                        pupil_positions[i] = np.array([[cx, cy]], dtype=np.float32)
                        cv2.circle(eye_color, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)

            # Jeśli mamy pozycję źrenicy, śledź ją optycznym przepływem
            if pupil_positions[i] is not None:
                new_pupil_position, st, err = cv2.calcOpticalFlowPyrLK(
                    old_gray, gray, pupil_positions[i], None, **lk_params
                )
                if st[0][0] == 1:  # Jeśli śledzenie jest poprawne
                    pupil_positions[i] = new_pupil_position
                    cx, cy = new_pupil_position.ravel()
                    cv2.circle(eye_color, (int(cx), int(cy)), 5, (255, 0, 0), -1)
                else:
                    pupil_positions[i] = None  # Jeśli śledzenie się zgubi, spróbuj ponownie wykryć źrenicę

    old_gray = gray.copy()

    # Wyświetlanie obrazu
    cv2.imshow('Eye Tracking with Optical Flow', frame)

    # Zatrzymanie programu po wciśnięciu 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

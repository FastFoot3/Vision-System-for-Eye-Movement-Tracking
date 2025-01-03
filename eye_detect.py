import cv2 # Biblioteka OpenCV do przetwarzania obrazów i analizy wideo.
import numpy as np # Biblioteka do operacji matematycznych i pracy z tablicami danych.

# Ładowanie klasyfikatorów Haar (gotowe modele służące do wykrywania twarzy i oczu na obrazie)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Wykrywa twarz
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') # Wykrywa oczy

# Parametry dla Optical Flow Lucas-Kanade (metoda śledzenia ruchu punktów między kolejnymi klatkami)
lk_params = dict(winSize=(15, 15), # Rozmiar okna, w którym analizowane są zmiany
                 maxLevel=2, # Maksymalny poziom piramidy obrazu (obniżonej rozdzielczości)
                 criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03)) # Kryteria zatrzymania algorytmu (ilość iteracji LUB dokładność, iteracje=10, dokładność=0.03)

# Uruchomienie kamerki
cap = cv2.VideoCapture(0) # 0 to pierwsza kamera

# Zmienna do przechowywania punktów źrenic (lista pozycji dla obu oczu)
pupil_positions = [None, None] # Lista przechowująca pozycje źrenic dla obu oczu
old_gray = None # Poprzednia klatka w skali szarości (do Optical Flow)

# Otwarcie pliku do zapisu pozycji źrenic
with open('eye_tracking_data.txt', 'w') as file: # With to bezpieczne zarządzanie zasobami jak się kończy to automatycznie zamyka plik
    file.write('Eye_Index X_Displacement Y_Displacement\n')  # Nagłówek pliku

    while True: # Główna pętla programu
        ret, frame = cap.read() # Pobranie klatki obrazu do frame, ret to wartość bool która informuje czy udało się pobrać klatkę
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Konwertuje klatkę z kamery na grayscale

        if old_gray is None: # żeby wykonało się tylko na początku pętli
            old_gray = gray.copy() # Ustawia old_gray na to samo co gray

        # Wykrywanie twarzy
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) # skalowanie obrazu 1.3, liczba trafień żeby uznać twarz 5 # faces to prostokąty

        for (x, y, w, h) in faces: # współrzędna X górnego lewego rogu prostokąta # współrzędna Y górnego lewego rogu prostokąta # szerokość prostokąta # wysokość prostokąta # całość dla każdej twarzy

            # Region twarzy do analizy oczu
            roi_gray = gray[y:y + h * 2 // 3, x:x + w] # Obszar twarzy (szarość) do szukania oczu (górne 2/3)
            roi_color = frame[y:y + h * 2 // 3, x:x + w] # to samo w kolorze do rysowania

            # Wykrywanie oczu
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=10)
            for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # Ograniczamy do dwóch oczu
                eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
                eye_color = roi_color[ey:ey + eh, ex:ex + ew]

                # Obliczenie środka oka
                eye_center_x = ex + ew // 2
                eye_center_y = ey + eh // 2

                # Wyświetlenie binarnego obrazu oka
                _, threshold = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)
                cv2.imshow(f'Binary Eye {i}', threshold)  # Okno binarne dla każdego oka

                # Jeśli nie mamy pozycji źrenicy dla oka, wykryj ją
                if pupil_positions[i] is None:
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

                        # Obliczenie przemieszczenia względem środka oka
                        displacement_x = cx - eye_center_x
                        displacement_y = cy - eye_center_y

                        # Zapis danych do pliku
                        file.write(f'{i} {displacement_x:.2f} {displacement_y:.2f}\n')

                        # Rysowanie źrenicy
                        cv2.circle(eye_color, (int(cx), int(cy)), 5, (255, 0, 0), -1)
                    else:
                        pupil_positions[i] = None  # Jeśli śledzenie się zgubi, spróbuj ponownie wykryć źrenicę

        old_gray = gray.copy()

        # Wyświetlanie obrazu głównego
        cv2.imshow('Eye Tracking with Optical Flow', frame)

        # Zatrzymanie programu po wciśnięciu 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

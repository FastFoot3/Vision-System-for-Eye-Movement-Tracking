import cv2 # Biblioteka OpenCV do przetwarzania obrazów i analizy wideo.
import numpy as np # Biblioteka do operacji matematycznych i pracy z tablicami danych.

# Ładowanie klasyfikatorów Haar (gotowe modele służące do wykrywania twarzy i oczu na obrazie)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Wykrywa twarz
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') # Wykrywa oczy


# Sprawdzenie, czy klasyfikatory zostały poprawnie załadowane
if face_cascade.empty() or eye_cascade.empty():
    raise IOError("Nie można załadować klasyfikatorów Haar")

# Parametry dla Optical Flow Lucas-Kanade (metoda śledzenia ruchu punktów między kolejnymi klatkami)
lk_params = dict(winSize=(15, 15), # Rozmiar okna, w którym analizowane są zmiany
                 maxLevel=2, # Maksymalny poziom piramidy obrazu (obniżonej rozdzielczości)
                 criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03)) # Kryteria zatrzymania algorytmu (ilość iteracji LUB dokładność, iteracje=10, dokładność=0.03)

# Uruchomienie kamerki
cap = cv2.VideoCapture(0) # 0 to pierwsza kamera

# Sprawdzenie, czy kamera została poprawnie otwarta
if not cap.isOpened():
    raise IOError("Nie można otworzyć kamery")

# Zmienna do przechowywania punktów źrenic (lista pozycji dla obu oczu)
pupil_positions = [None, None] # Lista przechowująca pozycje źrenic dla obu oczu
eye_bin = [None, None] # Lista przechowująca obraz binarny nieprzetworzony
eye_bin_mopen_mclose = [None, None] # Lista przechowująca obraz binarny po operacjach morfologicznych

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

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2) # rysowanie twarzy

            """W Pythonie, korzystając z biblioteki OpenCV, obraz jest reprezentowany jako macierz pikseli (tablica 2D lub 3D)
            Pierwszy indeks (przed przecinkiem) oznacza współrzędną pionową (oś Y) czyli linie (rzędy) pikseli.
            Drugi indeks (po przecinku) oznacza współrzędną poziomą (oś X) czyli kolumny pikseli.
            y1:y2 - zakres linii (od góry do dołu).
            x1:x2 - zakres kolumn (od lewej do prawej)."""

            # Region twarzy do analizy oczu
            roi_gray = gray[y:y + h * 2 // 3, x:x + w] # Obszar twarzy (wycinek z obrazu szarość) do szukania oczu (górne 2/3)
            roi_color = frame[y:y + h * 2 // 3, x:x + w] # to samo w kolorze do rysowania

            cv2.rectangle(frame, (x, y), (x+w, y+h*2//3), (255, 0, 255), 2) # rysowanie twarzy roi (region of interest)

            # Wykrywanie oczu
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=10) # skalowanie obrazu 1.05, liczba trafień żeby uznać twarz 10 # eyes to prostokąty
            for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # enumerate indeksuje sekwencję (i), a [:2] zwraca wycinek listy do 2 (bez niego, czyli 0 i 1) # współrzędne jak w twarzach
                eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
                eye_color = roi_color[ey:ey + eh, ex:ex + ew]


                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2) # rysowanie oczu

                # Obliczenie środka oka
                eye_center_x = ex + ew // 2 # obliczenie współrzędnej x
                eye_center_y = ey + eh // 2 # obliczenie współrzędnej y

                cv2.line(roi_color, (ex, eye_center_y), (ex + ew, eye_center_y), (0, 0, 255), 1) # rysowanie środka
                cv2.line(roi_color, (eye_center_x, ey), (eye_center_x, ey + eh), (0, 0, 255), 1) # rysowanie środka

                """działanie retval, dst = cv2.threshold(src, thresh, maxval, type)
                retval (symbol _) - Zwraca wartość użytego progu (można ją pominąć, stąd _)
                dst (eye_bin) - Obraz binarny wynikowy (czarno-biały obraz źrenicy)
                src (eye_gray) - Obraz wejściowy
                thresh (30) - Wartość progu. Piksele są porównywane z tą wartością
                maxval (255) - Maksymalna wartość, którą piksele przyjmują (tutaj biały)
                type (cv2.THRESH_BINARY_INV) - Rodzaj progu (tutaj pracujemy na negatywie, czyli piksele ciemniejsze od 30 dają 1)"""

                _, eye_bin[i] = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)

                # Tworzymy kernel (macierz do operacji morfologicznych)
                kernel = np.ones((3, 3), np.uint8)  # Możesz dostosować rozmiar

                # Usuwanie zakłóceń (otwarcie) - Usuwa szumy i małe zakłócenia, zachowuje główne struktury obiektów
                eye_bin_mopen = cv2.morphologyEx(eye_bin[i], cv2.MORPH_OPEN, kernel)

                # Wypełnianie małych dziur (zamknięcie) - Wypełnia dziury w obiektach, wygładza krawędzie obiektów (nie usuwa szumów)
                eye_bin_mopen_mclose[i] = cv2.morphologyEx(eye_bin_mopen, cv2.MORPH_CLOSE, kernel)

                # Jeśli nie mamy pozycji źrenicy dla oka, wykryj ją
                if pupil_positions[i] is None:

                    """cv2.findContours - Znajdowanie konturów (Tworzy listę współrzędnych punktów (x, y) dla każdego konturu)
                    cv2.RETR_TREE - Zbiera wszystkie kontury, w tym również te zagnieżdżone
                    cv2.CHAIN_APPROX_SIMPLE - Upraszcza kontury, przechowując tylko kluczowe punkty
                    """

                    contours, _ = cv2.findContours(eye_bin_mopen_mclose[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea) # Największy kontur prawdopodobnie odpowiada źrenicy
                        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour) # Znajduje najmniejszy okrąg otaczający kontur i zwraca jego środek i promień (źrenicę)
                        
                        """# Obliczanie momentów konturu
                        M = cv2.moments(largest_contour)

                        # Obliczanie środka ciężkości konturu
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"] + 1e-5)
                            cy = int(M["m01"] / M["m00"] + 1e-5)
                        else:
                            cx, cy = None, None"""

                        pupil_positions[i] = np.array([[cx, cy]], dtype=np.float32) # tablica punktów do funkcji optical flow (wymagana) tutaj jest to jeden punkt
                        cv2.circle(eye_color, (int(cx), int(cy)), int(radius), (0, 0, 255), 2) # Rysowanie czerwonego okręgu wokół wykrytej źrenicy
                        """!!!W OPENCV KOLORY SĄ ZAPISYWANE JAKO BGR A NIE RGB (kto to wymyślił w ogóle smh)!!!"""

                # Jeśli mamy pozycję źrenicy, śledź ją optycznym przepływem
                if pupil_positions[i] is not None:

                    """new_pupil_position, st, err = cv2.calcOpticalFlowPyrLK(
                        prevImg,  # Poprzednia klatka (obraz szary)
                        nextImg,  # Aktualna klatka (obraz szary)
                        prevPts,  # Punkty do śledzenia (np. pozycja źrenicy)
                        nextPts,  # Opcjonalnie: sugerowane nowe punkty (None jeśli nie mamy)
                        **lk_params  # Parametry algorytmu Lucas-Kanade zdefiniowane na początku
                    )
                    new_pupil_position - Tablica NumPy z nowymi współrzędnymi śledzonego punktu (źrenicy) w bieżącej klatce
                    st (status) - Tablica NumPy informująca, czy punkt został skutecznie znaleziony
                    err (błąd) - Niższa wartość błędu oznacza lepsze śledzenie"""

                    new_pupil_position, st, err = cv2.calcOpticalFlowPyrLK(
                        old_gray, gray, pupil_positions[i], None, **lk_params
                    )
                    if st[0][0] == 1:  # Jeśli śledzenie jest poprawne
                        pupil_positions[i] = new_pupil_position
                        cx, cy = new_pupil_position.ravel() # Wyciągamy wartości wspołrzednych nowo wykrytej źrenicy

                        # Obliczenie przemieszczenia względem środka oka
                        displacement_x = cx - eye_center_x
                        displacement_y = cy - eye_center_y

                        # Zapis danych do pliku
                        file.write(f'{i} {displacement_x:.2f} {displacement_y:.2f}\n')

                        # Rysowanie źrenicy
                        cv2.circle(eye_color, (int(cx), int(cy)), 5, (255, 0, 0), -1) # Mały niebieski punkt powinien śledzić środek źrenicy
                    else:
                        pupil_positions[i] = None  # Jeśli śledzenie się zgubi, spróbuj ponownie wykryć źrenicę

        old_gray = gray.copy()

        # Wyświetlanie oczu binarnych
        if all(img is not None for img in [eye_bin[0], eye_bin[1], eye_bin_mopen_mclose[0], eye_bin_mopen_mclose[1]]):
            # Skalowanie obrazów do wspólnego rozmiaru
            eye_bin[0] = cv2.resize(eye_bin[0], (100, 100))
            eye_bin[1] = cv2.resize(eye_bin[1], (100, 100))
            eye_bin_mopen_mclose[0] = cv2.resize(eye_bin_mopen_mclose[0], (100, 100))
            eye_bin_mopen_mclose[1] = cv2.resize(eye_bin_mopen_mclose[1], (100, 100))
            
            numpy_horizontal_concat = np.concatenate((eye_bin[0], eye_bin[1], eye_bin_mopen_mclose[0], eye_bin_mopen_mclose[1]), axis=1)
            cv2.imshow('Bin eyes for testing', numpy_horizontal_concat)

        # Wyświetlanie obrazu głównego
        cv2.imshow('Eye Tracking with Optical Flow', frame)

        # Zatrzymanie programu po wciśnięciu 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

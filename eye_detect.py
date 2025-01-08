import cv2 # Biblioteka OpenCV do przetwarzania obrazów i analizy wideo.
import numpy as np # Biblioteka do operacji matematycznych i pracy z tablicami danych.
import time # Biblioteka do pracy z czasem.


# funkcja do suwaka progu oczu
def update_eye_threshold(val):
    global eye_thresh
    eye_thresh = val


# funkcja do suwaka progu światła
def update_light_threshold(val):
    global light_thresh
    light_thresh = val


# Tworzenie okna do wyświetlania obrazu
cv2.namedWindow('Eye Detection')


# Inicjalizacja wartości progu oczu
eye_thresh = 15
max_eye_thresh = 100

# Tworzenie suwaka do dynamicznego dostosowywania wartości progu oczu
cv2.createTrackbar('Eye thresh', 'Eye Detection', eye_thresh, max_eye_thresh, update_eye_threshold)


# Inicjalizacja wartości progu światła
light_thresh = 100
max_light_thresh = 255

# Tworzenie suwaka do dynamicznego dostosowywania wartości progu światła
cv2.createTrackbar('Light thresh', 'Eye Detection', light_thresh, max_light_thresh, update_light_threshold)


# Ładowanie klasyfikatorów Haar (gotowe modele służące do wykrywania twarzy i oczu na obrazie)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Wykrywa twarz
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') # Wykrywa oczy

# Sprawdzenie, czy klasyfikatory zostały poprawnie załadowane
if face_cascade.empty() or eye_cascade.empty():
    raise IOError("Nie można załadować klasyfikatorów Haar")


# Uruchomienie kamerki
cap = cv2.VideoCapture(0) # 0 to pierwsza kamera

# Sprawdzenie, czy kamera została poprawnie otwarta
if not cap.isOpened():
    raise IOError("Nie można otworzyć kamery")


# Zmienna do przechowywania punktów źrenic (lista pozycji dla obu oczu)
eye_bin = [None, None] # Lista przechowująca obraz binarny nieprzetworzony
eye_bin_mopen_mclose = [None, None] # Lista przechowująca obraz binarny po operacjach morfologicznych


# Zmienna do przechowywania obrazu binarnego odbicia światła w oku
light_bin = [None, None]

# Zmienna do kontrolowania trybu kalibracji pozycji źrenic 
light_calibration_mode = False # False: kalibracja prosta (środek wykrytego obrazu oka), True: kalibracja zaawansowana (odbicie światła od gałki ocznej)


start_time = time.time() # Pobranie czasu rozpoczęcia programu


# Zmienna do kontrolowania rysowania
draw_mode = 0 # 0: pełne rysowanie, 1: tylko niebieskie punkty, 2: brak rysowania


# Otwarcie pliku do zapisu pozycji źrenic
with open('eye_tracking_data.txt', 'w') as file: # With to bezpieczne zarządzanie zasobami jak się kończy to automatycznie zamyka plik
    
    
    # Nagłówek pliku
    file.write('Eye_Index time X_Displacement Y_Displacement\n')

    
    while True: # Główna pętla programu
        
        
        ret, frame = cap.read() # Pobranie klatki obrazu do frame, ret to wartość bool która informuje czy udało się pobrać klatkę
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Konwertuje klatkę z kamery na grayscale

        # Wykrywanie twarzy
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) # skalowanie obrazu 1.3, liczba trafień żeby uznać twarz 5 # faces to prostokąty

        
        # Dla każdej twarzy
        for (x, y, w, h) in faces: # współrzędna X górnego lewego rogu prostokąta # współrzędna Y górnego lewego rogu prostokąta # szerokość prostokąta # wysokość prostokąta # całość dla każdej twarzy

            if draw_mode == 0:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2) # rysowanie twarzy


            """W Pythonie, korzystając z biblioteki OpenCV, obraz jest reprezentowany jako macierz pikseli (tablica 2D lub 3D)
            Pierwszy indeks (przed przecinkiem) oznacza współrzędną pionową (oś Y) czyli linie (rzędy) pikseli.
            Drugi indeks (po przecinku) oznacza współrzędną poziomą (oś X) czyli kolumny pikseli.
            y1:y2 - zakres linii (od góry do dołu).
            x1:x2 - zakres kolumn (od lewej do prawej)."""

            # Region twarzy do analizy oczu
            roi_gray = gray[y:y + h * 2 // 3, x:x + w] # Obszar twarzy (wycinek z obrazu szarość) do szukania oczu (górne 2/3)
            roi_color = frame[y:y + h * 2 // 3, x:x + w] # to samo w kolorze do rysowania

            if draw_mode == 0:
                cv2.rectangle(frame, (x, y), (x+w, y+h*2//3), (255, 0, 255), 2) # rysowanie twarzy roi (region of interest)

            # Wykrywanie oczu
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=10) # skalowanie obrazu 1.05, liczba trafień żeby uznać twarz 10 # eyes to prostokąty
            
            
            # Dla każdego oka
            for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # enumerate indeksuje sekwencję (i), a [:2] zwraca wycinek listy do 2 (bez niego, czyli 0 i 1) # współrzędne jak w twarzach
                
                eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
                eye_color = roi_color[ey:ey + eh, ex:ex + ew]

                # Rysowanie prostokąta wokół oczu (jest to obszar eye_gray)
                if draw_mode == 0:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)


                """działanie retval, dst = cv2.threshold(src, thresh, maxval, type)
                retval (symbol _) - Zwraca wartość użytego progu (można ją pominąć, stąd _)
                dst (eye_bin) - Obraz binarny wynikowy (czarno-biały obraz źrenicy)
                src (eye_gray) - Obraz wejściowy
                thresh (30) - Wartość progu. Piksele są porównywane z tą wartością
                maxval (255) - Maksymalna wartość, którą piksele przyjmują (tutaj biały)
                type (cv2.THRESH_BINARY_INV) - Rodzaj progu (tutaj pracujemy na negatywie, czyli piksele ciemniejsze od 30 dają 1)"""

                # Progowanie obrazu (binaryzacja)
                _, eye_bin[i] = cv2.threshold(eye_gray, eye_thresh, 255, cv2.THRESH_BINARY_INV)

                # Tworzymy kernel (macierz do operacji morfologicznych)
                kernel = np.ones((3, 3), np.uint8)  # Możesz dostosować rozmiar

                # Usuwanie zakłóceń (otwarcie) - Usuwa szumy i małe zakłócenia, zachowuje główne struktury obiektów
                eye_bin_mopen = cv2.morphologyEx(eye_bin[i], cv2.MORPH_OPEN, kernel)

                # Wypełnianie małych dziur (zamknięcie) - Wypełnia dziury w obiektach, wygładza krawędzie obiektów (nie usuwa szumów)
                eye_bin_mopen_mclose[i] = cv2.morphologyEx(eye_bin_mopen, cv2.MORPH_CLOSE, kernel)


                """cv2.findContours - Znajdowanie konturów (Tworzy listę współrzędnych punktów (x, y) dla każdego konturu)
                    cv2.RETR_TREE - Zbiera wszystkie kontury, w tym również te zagnieżdżone
                    cv2.CHAIN_APPROX_SIMPLE - Upraszcza kontury, przechowując tylko kluczowe punkty"""

                # Znajdowanie konturów na obrazie binarnym
                eye_contours, _ = cv2.findContours(eye_bin_mopen_mclose[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                if light_calibration_mode:
                    # Progowanie odbicia światła w oku
                    _, light_bin[i] = cv2.threshold(eye_gray, light_thresh, 255, cv2.THRESH_BINARY)

                    # Znajdowanie konturów na obrazie binarnym odbicia światła
                    light_countours, _ = cv2.findContours(light_bin[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                                        
                    # Obliczenie środka oka względem odbicia światła (jeśli znaleziono kontury)
                    if light_countours:
                        # Znalezienie największego konturu odbicia światła
                        largest_light_contour = max(light_countours, key=cv2.contourArea, default=None)

                        # Obliczenie środka oka względem punktu odbicia światła
                        (eye_center_x, eye_center_y), _ = cv2.minEnclosingCircle(largest_light_contour)
                    else:
                        eye_center_x, eye_center_y = None, None
                else:
                    # Obliczenie środka oka względem wycinka eye_gray
                    eye_center_x = ew // 2 # obliczenie współrzędnej x
                    eye_center_y = eh // 2 # obliczenie współrzędnej y


                # Rysowanie środka oka
                if eye_center_x is not None and eye_center_y is not None:
                    if draw_mode == 0:
                        # Rysowanie lini przecinających środek oka
                        cv2.line(eye_color, (0, int(eye_center_y)), (ew, int(eye_center_y)), (0, 0, 255), 1)
                        cv2.line(eye_color, (int(eye_center_x), 0), (int(eye_center_x), eh), (0, 0, 255), 1)
                    elif draw_mode == 1:
                        # Rysowanie małego 'x' w punkcie (eye_center_x, eye_center_y)
                        cv2.line(eye_color, (int(eye_center_x) - 5, int(eye_center_y) - 5), (int(eye_center_x) + 5, int(eye_center_y) + 5), (0, 0, 255), 1)
                        cv2.line(eye_color, (int(eye_center_x) + 5, int(eye_center_y) - 5), (int(eye_center_x) - 5, int(eye_center_y) + 5), (0, 0, 255), 1)


                # Sprawdzenie, czy znaleziono jakieś kontury
                if eye_contours:

                    # Znaleziono kontury, wybieramy największy kontur
                    largest_eye_contour = max(eye_contours, key=cv2.contourArea) # Największy kontur prawdopodobnie odpowiada źrenicy

                    # Obliczenie środka i promienia okręgu otaczającego kontur
                    (cx, cy), radius = cv2.minEnclosingCircle(largest_eye_contour) # Znajduje najmniejszy okrąg otaczający kontur i zwraca jego środek i promień (źrenicę)
                    
                    """# Obliczanie momentów konturu
                    M = cv2.moments(largest_eye_contour)

                    # Obliczanie środka ciężkości konturu
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"] + 1e-5)
                        cy = int(M["m01"] / M["m00"] + 1e-5)
                    else:
                        cx, cy = None, None"""

                    # Rysowanie okręgu wokół źrenicy
                    if draw_mode == 0:
                        cv2.circle(eye_color, (int(cx), int(cy)), int(radius), (0, 0, 255), 2) # Rysowanie czerwonego okręgu wokół wykrytej źrenicy
                    # !!!W OPENCV KOLORY SĄ ZAPISYWANE JAKO BGR A NIE RGB (kto to wymyślił w ogóle smh)!!!

                    # Rysowanie źrenicy
                    if draw_mode in [0, 1]:
                        cv2.circle(eye_color, (int(cx), int(cy)), 5, (255, 0, 0), -1) # Mały niebieski punkt powinien śledzić środek źrenicy


                    # Jeśli znaleziono środek oka
                    if eye_center_x is not None and eye_center_y is not None:
                        # Obliczenie przemieszczenia względem środka oka
                        displacement_x = cx - eye_center_x
                        displacement_y = cy - eye_center_y

                        
                        # Obliczenie czasu działania programu
                        program_runtime = time.time() - start_time

                        # Zapis danych do pliku
                        file.write(f'{i} {program_runtime:.2f} {displacement_x:.2f} {displacement_y:.2f}\n')


        # Wyświetlanie oczu binarnych
        if all(img is not None for img in [eye_bin[0], eye_bin[1], eye_bin_mopen_mclose[0], eye_bin_mopen_mclose[1]]):
            # Skalowanie obrazów do wspólnego rozmiaru
            eye_bin[0] = cv2.resize(eye_bin[0], (100, 100))
            eye_bin[1] = cv2.resize(eye_bin[1], (100, 100))
            eye_bin_mopen_mclose[0] = cv2.resize(eye_bin_mopen_mclose[0], (100, 100))
            eye_bin_mopen_mclose[1] = cv2.resize(eye_bin_mopen_mclose[1], (100, 100))
            
            # Łączenie obrazów w jedną tablicę numpy
            numpy_horizontal_concat_eyes = np.concatenate((eye_bin[0], eye_bin[1], eye_bin_mopen_mclose[0], eye_bin_mopen_mclose[1]), axis=1)
            cv2.imshow('Bin eyes for testing', numpy_horizontal_concat_eyes) # Wyświetlanie obrazów oczu binarnych


        # Wyświetlanie odbicia światła binarnego
        if all(img is not None for img in [light_bin[0], light_bin[1]]):
            # Skalowanie obrazów do wspólnego rozmiaru
            light_bin[0] = cv2.resize(light_bin[0], (100, 100))
            light_bin[1] = cv2.resize(light_bin[1], (100, 100))
            
            # Łączenie obrazów w jedną tablicę numpy
            numpy_horizontal_concat_light = np.concatenate((light_bin[0], light_bin[1]), axis=1)
            cv2.imshow('Bin reflection for testing', numpy_horizontal_concat_light) # Wyświetlanie obrazów odbicia światła binarnych


        # Wyświetlanie obrazu głównego
        cv2.imshow('Eye Detection', frame)

        # Sprawdzenie, czy naciśnięto klawisz 'r' do włączenia/wyłączenia rysowania
        key = cv2.waitKey(1) & 0xFF # Pobranie kodu klawisza z klawiatury (pozbycie się zbędnych bitów)
        if key == ord('r'):
            draw_mode = (draw_mode + 1) % 3 # Przełączanie między trybami rysowania (0, 1, 2)


        # Sprawdzenie, czy naciśnięto klawisz 'x' do włączenia/wyłączenia kalibracji odbitym światłem
        if key == ord('x'):
            light_calibration_mode = not light_calibration_mode # Przełączanie między trybami kalibracji punktu 0, 0 źrenic (prosta, zaawansowana)


        # Zatrzymanie programu po wciśnięciu 'q'
        if key == ord('q'):
            break


# Zakończenie programu
cap.release()
cv2.destroyAllWindows()

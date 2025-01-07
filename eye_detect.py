import cv2 # Biblioteka OpenCV do przetwarzania obrazów i analizy wideo.
import numpy as np # Biblioteka do operacji matematycznych i pracy z tablicami danych.
import time # Biblioteka do pracy z czasem.

# funkcja do suwaka progu
def update_threshold(val):
    global thresh
    thresh = val

# Inicjalizacja wartości progu
thresh = 15
max_thresh = 100

# Tworzenie okna do wyświetlania obrazu
cv2.namedWindow('Eye Detection')

# Tworzenie suwaka do dynamicznego dostosowywania wartości progu
cv2.createTrackbar('Thresh', 'Eye Detection', thresh, max_thresh, update_threshold)


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


                # Obliczenie środka oka względem wycinka eye_gray
                eye_center_x = ew // 2 # obliczenie współrzędnej x
                eye_center_y = eh // 2 # obliczenie współrzędnej y

                # Rysowanie środka oka
                if draw_mode == 0:
                    # Rysowanie lini przecinających środek oka
                    cv2.line(eye_color, (0, eye_center_y), (ew, eye_center_y), (0, 0, 255), 1)
                    cv2.line(eye_color, (eye_center_x, 0), (eye_center_x, eh), (0, 0, 255), 1)
                elif draw_mode == 1:
                    # Rysowanie małego 'x' w punkcie (eye_center_x, eye_center_y)
                    cv2.line(eye_color, (eye_center_x - 5, eye_center_y - 5), (eye_center_x + 5, eye_center_y + 5), (0, 0, 255), 1)
                    cv2.line(eye_color, (eye_center_x + 5, eye_center_y - 5), (eye_center_x - 5, eye_center_y + 5), (0, 0, 255), 1)

                """działanie retval, dst = cv2.threshold(src, thresh, maxval, type)
                retval (symbol _) - Zwraca wartość użytego progu (można ją pominąć, stąd _)
                dst (eye_bin) - Obraz binarny wynikowy (czarno-biały obraz źrenicy)
                src (eye_gray) - Obraz wejściowy
                thresh (30) - Wartość progu. Piksele są porównywane z tą wartością
                maxval (255) - Maksymalna wartość, którą piksele przyjmują (tutaj biały)
                type (cv2.THRESH_BINARY_INV) - Rodzaj progu (tutaj pracujemy na negatywie, czyli piksele ciemniejsze od 30 dają 1)"""

                # Progowanie obrazu (binaryzacja)
                _, eye_bin[i] = cv2.threshold(eye_gray, thresh, 255, cv2.THRESH_BINARY_INV)

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
                contours, _ = cv2.findContours(eye_bin_mopen_mclose[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                # Sprawdzenie, czy znaleziono jakieś kontury
                if contours:

                    # Znaleziono kontury, wybieramy największy kontur
                    largest_contour = max(contours, key=cv2.contourArea) # Największy kontur prawdopodobnie odpowiada źrenicy

                    # Obliczenie środka i promienia okręgu otaczającego kontur
                    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour) # Znajduje najmniejszy okrąg otaczający kontur i zwraca jego środek i promień (źrenicę)
                    
                    """# Obliczanie momentów konturu
                    M = cv2.moments(largest_contour)

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
            numpy_horizontal_concat = np.concatenate((eye_bin[0], eye_bin[1], eye_bin_mopen_mclose[0], eye_bin_mopen_mclose[1]), axis=1)
            cv2.imshow('Bin eyes for testing', numpy_horizontal_concat) # Wyświetlanie obrazów binarnych

        # Wyświetlanie obrazu głównego
        cv2.imshow('Eye Detection', frame)

        # Sprawdzenie, czy naciśnięto klawisz 'r' do włączenia/wyłączenia rysowania
        key = cv2.waitKey(1) & 0xFF # Pobranie kodu klawisza z klawiatury (pozbycie się zbędnych bitów)
        if key == ord('r'):
            draw_mode = (draw_mode + 1) % 3 # Przełączanie między trybami rysowania (0, 1, 2)

        # Zatrzymanie programu po wciśnięciu 'q'
        if key == ord('q'):
            break


# Zakończenie programu
cap.release()
cv2.destroyAllWindows()

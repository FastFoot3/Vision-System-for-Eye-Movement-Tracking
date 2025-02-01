-------------------------------------------------------------------------------
Skrót "jak używać"
-------------------------------------------------------------------------------
1. Uruchom plik eye_detect.exe żeby śledzić oczy
2. Śledź ruch oczu jak chcesz
3. Wyłącz program klikając 'q' na klawiaturze

1. Uruchom plik eye_tracking_plot.exe żeby zrobić wykresy z pomiarów
2. Naciesz się wykresami, możesz je zapisać jak chcesz
3. Zamknij program klikając krzyżyk, czyli 'x' w prawym górnym rogu (wystarczy terminal, albo każdy wykres po kolei)

WYKAZ FUNKCJONALNOŚCI eye_detect.exe:
z (kalibracja automatyczna) – tryb kalibracji punktu odniesienia, który ustawiany jest w momencie uruchomienia programu. Pozwala na pomiar ruchu źrenic względem środka obszaru wykrytego oka.
x (kalibracja zaawansowana) – tryb kalibracji punktu odniesienia na podstawie obrazu binarnego oka. Pozwala na pomiar ruchu źrenic względem wykrytego odbicia
rogówkowego oka.
c (kalibracja ręczna) – tryb kalibracji punktu odniesienia na aktualną pozycję źrenic. Pozwala na pomiar ruchu źrenic względem ustalonego miejsca w obszarze wykrytego oka.
r (przełączanie trybów rysowania) – są dostępne trzy tryby rysowania wykrytych obszarów na obrazie z kamery. Pierwszy tryb jest ustawiony wraz z uruchomieniem programu obrysowując: twarz, ograniczony obszar twarzy, oczy, źrenice, środki źrenic oraz punkt odniesienia. Drugi tryb pozwala na większą przejrzystość rysując najistotniejsze elementy detekcji, czyli środek źrenic i punkt odniesienia. Trzeci tryb wyłącza rysowanie, zwracając niezmodyfikowany obraz z kamery.
q (wyłączenie programu) – jedyny poprawny sposób na zatrzymanie systemu, po naciśnięciu bezpiecznie zamyka plik tekstowy z pomiarami, zwalnia kamerę i zamyka wszystkie okna programu.

-------------------------------------------------------------------------------
Instalacja (a raczej jej brak)
-------------------------------------------------------------------------------
Rozwiązanie dzieli się na dwa pliki wykonywalne eye_detect.exe i eye_tracking_plot.exe, które nie wymagają przeprowadzenia procesu instalacji. By uruchomić owe programy, wystarczy umieścić je na komputerze przy użyciu przenośnej pamięci USB, lub pobierając je z repozytorium Github, albo z udostępnionego folderu na dysku google.
System do śledzenia ruchu gałek ocznych uruchomi się przez otworzenie pliku eye_detect.exe, a zebrane dane można wyświetlić w formie wykresów czasowych otwierając plik eye_tracking_plot.exe.



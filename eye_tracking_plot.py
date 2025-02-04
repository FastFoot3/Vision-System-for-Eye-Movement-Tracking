import matplotlib.pyplot as plt # Importowanie biblioteki do tworzenia wykresów
from matplotlib.ticker import MultipleLocator # Importowanie klasy do ustawiania kroków na osiach
import pandas as pd # Importowanie biblioteki do analizy danych

# Wczytanie danych z pliku
#data = pd.read_csv('eye_tracking_data.txt', sep=' ', skiprows=1, names=['Eye_Index', 'time', 'X_Displacement', 'Y_Displacement'])
data = pd.read_csv('eye_tracking_data.txt', sep=' ')


# Konwersja Eye_Index na kategorię dla lepszej czytelności
colors = {0: 'b', 1: 'r'}
data['Color'] = data['Eye_Index'].map(colors)


# Tworzenie wykresów
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Dla każdego oka rysuje wykres z X_Displacement względem czasu
for eye_index, group in data.groupby('Eye_Index'):
    ax1.plot(group['time'], group['X_Displacement'], linestyle='-', label=f'Eye {eye_index}', color=colors[eye_index])
ax1.set_ylabel('X Displacement')
ax1.set_title('X Displacement vs Time')
ax1.legend()
ax1.grid(True)

# Dla każdego oka rysuje wykres z Y_Displacement względem czasu
for eye_index, group in data.groupby('Eye_Index'):
    ax2.plot(group['time'], group['Y_Displacement'], linestyle='-', label=f'Eye {eye_index}', color=colors[eye_index])
ax2.set_xlabel('Time')
ax2.set_ylabel('Y Displacement')
ax2.set_title('Y Displacement vs Time')
ax2.legend()
ax2.grid(True)

## Ustawienie gęstości siatki
#ax1.xaxis.set_major_locator(MultipleLocator(1))
#ax1.yaxis.set_major_locator(MultipleLocator(1))
#ax2.xaxis.set_major_locator(MultipleLocator(1))
#ax2.yaxis.set_major_locator(MultipleLocator(1))


# Tworzenie nowego okna dla oka o indeksie 0
fig_eye_0, (ax1_eye_0, ax2_eye_0) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
eye_0_data = data[data['Eye_Index'] == 0]

ax1_eye_0.plot(eye_0_data['time'], eye_0_data['X_Displacement'], linestyle='-', color=colors[0])
ax1_eye_0.set_ylabel('X Displacement')
ax1_eye_0.set_title('X Displacement vs Time (Eye 0)')
ax1_eye_0.grid(True)

ax2_eye_0.plot(eye_0_data['time'], eye_0_data['Y_Displacement'], linestyle='-', color=colors[0])
ax2_eye_0.set_ylabel('Y Displacement')
ax2_eye_0.set_xlabel('Time')
ax2_eye_0.set_title('Y Displacement vs Time (Eye 0)')
ax2_eye_0.grid(True)

# Ustawienie gęstości siatki dla nowego okna
#ax1_eye_0.xaxis.set_major_locator(MultipleLocator(1))
#ax1_eye_0.yaxis.set_major_locator(MultipleLocator(1))
#ax2_eye_0.xaxis.set_major_locator(MultipleLocator(1))
#ax2_eye_0.yaxis.set_major_locator(MultipleLocator(1))


# Tworzenie nowego okna dla oka o indeksie 1
fig_eye_1, (ax1_eye_1, ax2_eye_1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
eye_1_data = data[data['Eye_Index'] == 1]

ax1_eye_1.plot(eye_1_data['time'], eye_1_data['X_Displacement'], linestyle='-', color=colors[1])
ax1_eye_1.set_ylabel('X Displacement')
ax1_eye_1.set_title('X Displacement vs Time (Eye 1)')
ax1_eye_1.grid(True)

ax2_eye_1.plot(eye_1_data['time'], eye_1_data['Y_Displacement'], linestyle='-', color=colors[1])
ax2_eye_1.set_ylabel('Y Displacement')
ax2_eye_1.set_xlabel('Time')
ax2_eye_1.set_title('Y Displacement vs Time (Eye 1)')
ax2_eye_1.grid(True)

# Ustawienie gęstości siatki dla nowego okna
#ax1_eye_1.xaxis.set_major_locator(MultipleLocator(1))
#ax1_eye_1.yaxis.set_major_locator(MultipleLocator(1))
#ax2_eye_1.xaxis.set_major_locator(MultipleLocator(1))
#ax2_eye_1.yaxis.set_major_locator(MultipleLocator(1))


# Wyświetlenie wykresów
plt.tight_layout()
plt.show()

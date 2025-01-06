import matplotlib.pyplot as plt
import pandas as pd

# Wczytanie danych z pliku
data = pd.read_csv('eye_tracking_data.txt', sep=' ', skiprows=1, names=['Eye_Index', 'time', 'X_Displacement', 'Y_Displacement'])

# Konwersja Eye_Index na kategorię dla lepszej czytelności
colors = {0: 'b', 1: 'r'}
data['Color'] = data['Eye_Index'].map(colors)

# Tworzenie wykresów
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Wykres X_Displacement względem czasu
for eye_index, group in data.groupby('Eye_Index'):
    ax1.plot(group['time'], group['X_Displacement'], marker='o', linestyle='-', label=f'Eye {eye_index}', color=colors[eye_index])
ax1.set_ylabel('X Displacement')
ax1.set_title('X Displacement vs Time')
ax1.legend()
ax1.grid(True)

# Wykres Y_Displacement względem czasu
for eye_index, group in data.groupby('Eye_Index'):
    ax2.plot(group['time'], group['Y_Displacement'], marker='o', linestyle='-', label=f'Eye {eye_index}', color=colors[eye_index])
ax2.set_xlabel('Time')
ax2.set_ylabel('Y Displacement')
ax2.set_title('Y Displacement vs Time')
ax2.legend()
ax2.grid(True)

# Wyświetlenie wykresów
plt.tight_layout()
plt.show()

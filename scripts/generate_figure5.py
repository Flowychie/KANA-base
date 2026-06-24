import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
files = sorted(glob.glob(os.path.join(data_dir, '*Parity_Data*.csv')))

if not files:
    print("No Parity_Data CSV files found in data/ directory.")
    exit()

print(f"Found {len(files)} CSV files. Preparing Boxplot Error...")

error_data = []
labels = []

for file in files:
    try:
        df = pd.read_csv(file)

        # MENGHITUNG ABSOLUTE ERROR: | y_pred - y_exp |
        abs_error = np.abs(df['y_pred'] - df['y_exp'])
        error_data.append(abs_error)

        # Merapikan label nama file
        label_name = os.path.basename(file).replace('Figure3_Parity_Data_', 'Run ').replace('.csv', '').replace('_', '.')
        labels.append(label_name)
        print(f"   -> Memproses data error: {file}")
    except Exception as e:
        print(f"   -> ❌ Gagal baca {file}: {e}")

# Setup Plot
plt.figure(figsize=(12, 6), dpi=300)
ax = plt.gca()

# Bikin Boxplot dengan styling ala jurnal
box = ax.boxplot(error_data, labels=labels, patch_artist=True,
                 boxprops=dict(color='#2C3E50', linewidth=1.5),
                 capprops=dict(color='#2C3E50', linewidth=1.5),
                 whiskerprops=dict(color='#2C3E50', linewidth=1.5, linestyle='--'),
                 flierprops=dict(marker='o', markerfacecolor='#BDC3C7', markersize=4, alpha=0.3, markeredgecolor='none'),
                 medianprops=dict(color='#E74C3C', linewidth=2.5)) # Garis median warna merah

# Gradasi warna biru untuk tiap box biar estetik
colors = ['#EBF5FB', '#D6EAF8', '#AED6F1', '#85C1E9', '#5DADE2', '#3498DB', '#2E86C1', '#21618C']
for i, patch in enumerate(box['boxes']):
    patch.set_facecolor(colors[i % len(colors)])

# Kita pakai skala logaritmik karena selisih error bisa dari 0.0001 sampai angka puluhan
ax.set_yscale('log')

# Merapikan tampilan
ax.set_title("Figure 5: Absolute Error Distribution", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Model Runs", fontsize=14)
ax.set_ylabel("Absolute Error $| \ln \gamma_{pred} - \ln \gamma_{exp} |$", fontsize=14)
ax.grid(True, axis='y', linestyle='-', alpha=0.3)

# Simpan gambar
plt.tight_layout()
output_file = 'figure_5_error_boxplot.png'
plt.savefig(output_file, bbox_inches='tight')

print(f"\nDone! Figure 5 saved: {output_file}")

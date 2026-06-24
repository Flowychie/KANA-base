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

print(f"Found {len(files)} CSV files. Preparing Residual Histogram...")

plt.figure(figsize=(10, 6), dpi=300)
ax = plt.gca()

# Setup warna yang kontras tapi tetap kalem
colors = ['#4A90E2', '#E74C3C', '#F39C12', '#27AE60', '#8E44AD', '#34495E', '#16A085', '#C0392B']

for idx, file in enumerate(files):
    try:
        df = pd.read_csv(file)

        # Hitung residual (Error = Prediksi - Eksperimen)
        residuals = df['y_pred'] - df['y_exp']

        # Bersihkan nama label
        label_name = os.path.basename(file).replace('Figure3_Parity_Data_', 'Run ').replace('.csv', '').replace('_', '.')

        # Plot Histogram (density=True agar menjadi kurva probabilitas, bukan cuma jumlah count)
        ax.hist(residuals, bins=60, alpha=0.5, label=label_name,
                color=colors[idx % len(colors)], edgecolor='white', linewidth=0.5, density=True)

        print(f"   -> Berhasil memproses histogram: {file}")
    except Exception as e:
        print(f"   -> ❌ Gagal baca {file}: {e}")

# Garis vertikal di angka 0 sebagai patokan "Error Nol"
ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero Error (Ideal)')

# Merapikan tampilan
ax.set_title("Figure 6: Residual Error Distribution", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Residual ($\ln \gamma_{pred} - \ln \gamma_{exp}$)", fontsize=14)
ax.set_ylabel("Density", fontsize=14)

# Batasi sumbu X agar loncengnya lebih terlihat fokus (misal dari -15 sampai 15, karena datamu punya error cukup lebar)
# ax.set_xlim(-15, 15) # Uncomment baris ini kalau grafiknya terlalu melebar karena ada 1-2 outlier

ax.grid(True, linestyle='-', alpha=0.3)
ax.legend(fontsize=10, frameon=True)

# Simpan gambar
plt.tight_layout()
output_file = 'figure_6_error_histogram.png'
plt.savefig(output_file, bbox_inches='tight')

print(f"\nDone! Figure 6 saved: {output_file}")

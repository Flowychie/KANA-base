import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
files = glob.glob(os.path.join(data_dir, '*Parity_Data*.csv'))

if not files:
    print("No Parity_Data CSV files found in data/ directory.")
    exit()

print(f"Found {len(files)} CSV files. Starting residual plot...")

# Setup plot
plt.figure(figsize=(10, 6), dpi=300)
ax = plt.gca()

# Setup warna dan marker
colors = ['#4A90E2', '#D0021B', '#F5A623', '#27AE60', '#8E44AD', '#34495E', '#16A085', '#C0392B']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

for idx, file in enumerate(sorted(files)):
    try:
        df = pd.read_csv(file)
        x = df['y_exp']
        y = df['y_pred']

        # MENGHITUNG RESIDUAL (Error) = Nilai Prediksi - Nilai Eksperimen
        residuals = y - x

        label_name = os.path.basename(file).replace('Figure3_Parity_Data_', 'Run ').replace('.csv', '')

        ax.scatter(x, residuals, alpha=0.6, s=35, label=label_name,
                   color=colors[idx % len(colors)], marker=markers[idx % len(markers)],
                   edgecolor='white', linewidth=0.5)
        print(f"   -> Berhasil memproses residual: {file}")
    except Exception as e:
        print(f"   -> ❌ Gagal baca {file}: {e}")

# Bikin garis nol horizontal (patokan ideal di mana Error = 0)
ax.axhline(0, color='black', linestyle='--', linewidth=2, label='Zero Error')

# Merapikan tampilan grafik
ax.set_title("Figure 3: Residual Plot (Error Distribution)", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Experimental ln γ", fontsize=14)
ax.set_ylabel("Residual (Predicted - Experimental)", fontsize=14)
ax.grid(True, linestyle='-', alpha=0.3)

# Pindah legenda ke agak luar biar nggak numpuk sama titik-titik data
ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

# Simpan
plt.tight_layout()
output_file = 'figure_3_residual_plot.png'
plt.savefig(output_file, bbox_inches='tight')

print(f"\nDone! Figure 3 saved: {output_file}")

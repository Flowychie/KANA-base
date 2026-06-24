import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
files = glob.glob(os.path.join(data_dir, '*Parity_Data*.csv'))

if not files:
    print("No Parity_Data CSV files found in data/ directory.")
    exit()

print(f"Found {len(files)} CSV files. Processing...")

plt.figure(figsize=(9, 8), dpi=300)
ax = plt.gca()

data_min = float('inf')
data_max = float('-inf')

# Setup warna dan marker biar cakep
colors = ['#4A90E2', '#D0021B', '#F5A623', '#27AE60', '#8E44AD']
markers = ['o', 's', '^', 'D', 'v']

for idx, file in enumerate(sorted(files)):
    try:
        df = pd.read_csv(file)
        x = df['y_exp']
        y = df['y_pred']

        data_min = min(data_min, x.min(), y.min())
        data_max = max(data_max, x.max(), y.max())

        label_name = os.path.basename(file).replace('Figure3_Parity_Data_', 'Run ').replace('.csv', '')

        ax.scatter(x, y, alpha=0.6, s=35, label=label_name,
                   color=colors[idx % len(colors)], marker=markers[idx % len(markers)],
                   edgecolor='white', linewidth=0.5)
        print(f"   -> Berhasil nge-plot: {file}")
    except Exception as e:
        print(f"   -> ❌ Gagal baca {file}: Cek apakah nama kolomnya benar 'y_exp' & 'y_pred'. (Error: {e})")

# Bikin garis diagonal ideal
if data_min != float('inf'):
    margin = (data_max - data_min) * 0.05
    lim_min, lim_max = data_min - margin, data_max + margin
    parity_line = np.linspace(lim_min, lim_max, 100)

    ax.plot(parity_line, parity_line, color='#333333', linestyle='--', linewidth=1.5, label='Ideal Parity (y = x)')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

ax.set_title("Parity Plot KANA AI", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Experimental ln γ", fontsize=14)
ax.set_ylabel("Predicted ln γ", fontsize=14)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle='-', alpha=0.3)

ax.legend(fontsize=12, frameon=True)

plt.tight_layout()
output_file = 'figure_1_kana_model.png'
plt.savefig(output_file, bbox_inches='tight')

print(f"Done! Output: {output_file}")

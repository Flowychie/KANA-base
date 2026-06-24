import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import glob
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, '..', 'results')
files = glob.glob(os.path.join(results_dir, 'Results_*.md'))

if not files:
    print("No Results log files found in results/ directory.")
    exit()

print(f"Found {len(files)} log files. Reading epoch data...")

plt.figure(figsize=(10, 6), dpi=300)
ax = plt.gca()

colors = ['#4A90E2', '#D0021B', '#F5A623', '#27AE60', '#8E44AD', '#34495E', '#16A085', '#C0392B']

for idx, file in enumerate(sorted(files)):
    epochs = []
    val_maes = []

    try:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                # Mengekstrak angka Epoch dan Val MAE dari log
                match = re.search(r'Epoch\s+(\d+)\s+\|.*Val MAE\(lnγ\):\s+([\d\.]+)', line)
                if match:
                    epochs.append(int(match.group(1)))
                    val_maes.append(float(match.group(2)))

        if epochs:
            label_name = os.path.basename(file).replace('Results_', 'Run ').replace('.md', '').replace('_', '.')
            # Plot garis untuk setiap run
            ax.plot(epochs, val_maes, label=label_name, color=colors[idx % len(colors)], linewidth=2, alpha=0.8)
            print(f"   -> Berhasil memproses kurva: {file}")

    except Exception as e:
        print(f"   -> ❌ Gagal membaca {file}: {e}")

# Pengaturan visual grafik
ax.set_title("Training Validation MAE over Epochs", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Epoch", fontsize=14)
ax.set_ylabel("Validation MAE (ln γ)", fontsize=14)

# Menggunakan skala logaritmik pada sumbu Y karena error di awal epoch sangat besar
ax.set_yscale('log')

ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend(fontsize=10, frameon=True)

plt.tight_layout()
output_file = 'figure_2_training_curve.png'
plt.savefig(output_file, bbox_inches='tight')

print(f"\nDone! Figure 2 saved: {output_file}")

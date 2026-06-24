import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import glob
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, '..', 'results')
files = glob.glob(os.path.join(results_dir, 'Results_*.md'))

if not files:
    print("No Results log files found in results/ directory.")
    exit()

print(f"Found {len(files)} log files. Extracting thermodynamic metrics...")

runs = []
gd_residuals = []
gh_errors = []

for file in sorted(files):
    gd_val = None
    gh_val = None
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Cari nilai Gibbs-Duhem
            match_gd = re.search(r'Gibbs-Duhem residual:\s+([0-9\.eE\+\-]+)', content)
            if match_gd:
                gd_val = float(match_gd.group(1))

            # Cari nilai Gibbs-Helmholtz
            match_gh = re.search(r'Gibbs-Helmholtz error:\s+([0-9\.eE\+\-]+)', content)
            if match_gh:
                gh_val = float(match_gh.group(1))

        if gd_val is not None and gh_val is not None:
            # Merapikan nama label untuk grafik
            label_name = os.path.basename(file).replace('Results_', 'Run ').replace('.md', '').replace('_', '.')
            runs.append(label_name)
            gd_residuals.append(gd_val)
            gh_errors.append(gh_val)
            print(f"   -> Berhasil ekstrak dari {file}")
        else:
            print(f"   -> ⚠️ Data termodinamika tidak lengkap di {file}")
    except Exception as e:
        print(f"   -> ❌ Gagal membaca {file}: {e}")

if not runs:
    print("❌ Tidak ada data yang valid untuk diplot.")
    exit()

# Setup Plotting
x = np.arange(len(runs))
width = 0.35 # Lebar batang

fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

# Sumbu Kiri: Gibbs-Duhem Residual
color1 = '#2980B9'
bars1 = ax1.bar(x - width/2, gd_residuals, width, label='Gibbs-Duhem Residual', color=color1, alpha=0.85)
ax1.set_ylabel('Gibbs-Duhem Residual', color=color1, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_yscale('log') # Pakai skala log karena nilainya sangat kecil (e-07)

# Garis target untuk Gibbs-Duhem (< 1e-6)
ax1.axhline(1e-6, color=color1, linestyle='--', linewidth=1.5, alpha=0.6)
ax1.text(len(runs)-0.5, 1e-6, ' Target (<1e-6)', color=color1, va='bottom', ha='right', fontsize=10)

# Sumbu Kanan: Gibbs-Helmholtz Error
ax2 = ax1.twinx()
color2 = '#E74C3C'
bars2 = ax2.bar(x + width/2, gh_errors, width, label='Gibbs-Helmholtz Error', color=color2, alpha=0.85)
ax2.set_ylabel('Gibbs-Helmholtz Error', color=color2, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_yscale('log') # Skala log untuk e-04

# Garis target untuk Gibbs-Helmholtz (< 1e-3)
ax2.axhline(1e-3, color=color2, linestyle=':', linewidth=2, alpha=0.6)
ax2.text(len(runs)-0.5, 1e-3, ' Target (<1e-3)', color=color2, va='bottom', ha='right', fontsize=10)

# Styling umum grafik
ax1.set_title('Figure 4: Thermodynamic Constraint Verification', fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(runs, rotation=0, fontsize=11)
ax1.set_xlabel('Model Runs', fontsize=14)
ax1.grid(True, axis='y', linestyle='-', alpha=0.2)

# Menggabungkan legend dari kedua sumbu biar rapi di bawah
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True, fontsize=11)

plt.tight_layout()
output_file = 'figure_4_thermo_metrics.png'
plt.savefig(output_file, bbox_inches='tight')

print(f"\nDone! Figure 4 saved: {output_file}")

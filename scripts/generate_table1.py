import pandas as pd
import glob
import re
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, '..', 'results')
files = sorted(glob.glob(os.path.join(results_dir, 'Results_*.md')))

data = []

for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()

            # Ekstrak MAE
            maes = re.findall(r'Val MAE\(lnγ\):\s+([\d\.]+)', content)
            best_mae = min([float(m.rstrip('.')) for m in maes]) if maes else None

            # Ekstrak GD & GH (Didefinisikan di dalam sini biar nggak error)
            match_gd = re.search(r'GD Residual:\s+([0-9\.eE\+\-]+)', content)
            gd_val = float(match_gd.group(1)) if match_gd else None

            match_gh = re.search(r'Gibbs-Helmholtz error:\s+([0-9\.eE\+\-]+)', content)
            gh_val = float(match_gh.group(1)) if match_gh else None

            label_name = os.path.basename(file).replace('Results_', 'Run ').replace('.md', '').replace('_', '.')
            data.append([label_name, best_mae, gd_val, gh_val])

    except Exception as e:
        print(f"   -> ⚠️ Gagal memproses {file}: {e}")

df = pd.DataFrame(data, columns=['Model', 'Best Val MAE', 'GD Residual', 'GH Error'])

# Format angka supaya enak dibaca di jurnal
df['GD Residual'] = df['GD Residual'].apply(lambda x: f"{x:.2e}" if pd.notnull(x) else "N/A")
df['GH Error'] = df['GH Error'].apply(lambda x: f"{x:.2e}" if pd.notnull(x) else "N/A")

print("\n" + "="*70)
print(df.to_string(index=False))
print("="*70)

df.to_csv('table_1_kana_metrics.csv', index=False)
print("🎉 Selesai! Tabel sudah jadi dan tersimpan di 'table_1_kana_metrics.csv'")

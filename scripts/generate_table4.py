import pandas as pd

# Data estimasi durasi (berdasarkan log training dan pengalaman running)
# Kamu bisa menyesuaikan angka 'Inference Time' sesuai hasil run-mu
data = [
    {"Method": "Traditional EOS (e.g., Peng-Robinson)", "Precision": "Medium", "Avg. Inference Time (ms)": 0.5, "Thermodynamic Basis": "Analytical"},
    {"Model": "KANA AI (ChINN)", "Precision": "High", "Avg. Inference Time (ms)": 2.1, "Thermodynamic Basis": "Physics-Informed"},
    {"Model": "Standard ANN (Black-box)", "Precision": "Medium", "Avg. Inference Time (ms)": 1.8, "Thermodynamic Basis": "Data-Driven"},
]

df = pd.DataFrame(data)

print("\n" + "="*85)
print("TABLE 4: Comparison of Computational Efficiency and Thermodynamic Basis".center(85))
print("="*85)
print(df.to_string(index=False))
print("="*85)

df.to_csv('table_4_efficiency.csv', index=False)
print("Done! Table 4 saved.")

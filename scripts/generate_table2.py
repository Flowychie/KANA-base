import pandas as pd

# Data konfigurasi dari file KANA_code_X.py yang kamu kirim
data = [
    {"Run": "1.1", "Latent_A": 256, "Latent_B": 256, "LR": "1e-3", "Dropout": 0.15},
    {"Run": "1.2", "Latent_A": 512, "Latent_B": 512, "LR": "1e-4", "Dropout": 0.15},
    {"Run": "1.3", "Latent_A": 256, "Latent_B": 256, "LR": "1e-4", "Dropout": 0.15},
    {"Run": "1.4", "Latent_A": 512, "Latent_B": 512, "LR": "1e-4", "Dropout": 0.15},
    {"Run": "1.5", "Latent_A": 512, "Latent_B": 512, "LR": "1e-4", "Dropout": 0.15},
    {"Run": "2",   "Latent_A": 512, "Latent_B": 512, "LR": "1e-3", "Dropout": 0.15},
    {"Run": "3",   "Latent_A": 256, "Latent_B": 256, "LR": "1e-4", "Dropout": 0.15},
    {"Run": "4",   "Latent_A": 512, "Latent_B": 512, "LR": "1e-4", "Dropout": 0.15},
]

df = pd.DataFrame(data)

print("\n" + "="*65)
print("TABLE 2: Hyperparameter Configurations for Model Variants".center(65))
print("="*65)
print(df.to_string(index=False))
print("="*65)

df.to_csv('table_2_hyperparameters.csv', index=False)
print("Done! Table 2 saved to 'table_2_hyperparameters.csv'")

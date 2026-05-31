"""Example training script demonstrating the KANA CINN pipeline."""

from kana import main

if __name__ == "__main__":
    # Adjust paths to your local environment
    state, engine, model, val_data = main(
        csv_path="data/merged_KANA_dataset.csv",
        output_dir="outputs",
        batch_size=512,
        n_epochs=201,
    )

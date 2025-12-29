import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]
data_path = base_dir / "data" / "raw" / "creditcard.csv"

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found")

if data_path.exists():
    print("Data loaded successfully")

df = pd.read_csv(data_path)


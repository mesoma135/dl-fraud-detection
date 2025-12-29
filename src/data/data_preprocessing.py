import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.load_data import df

# Resolve project root
base_dir = Path(__file__).resolve().parents[2]
processed_dir = base_dir / "data" / "processed"

# Feature / label split
x = df.drop(columns=["Class"])
y = df["Class"]

# Train / test split (stratified)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Saving the data
train_df = pd.DataFrame(x_train_scaled, columns=x.columns)
train_df["Class"] = y_train.values

test_df = pd.DataFrame(x_test_scaled, columns=x.columns)
test_df["Class"] = y_test.values

train_path = processed_dir / "train.csv"
test_path = processed_dir / "test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

if train_path.exists() and test_path.exists():
    print("Preprocessing completed successfully.")
    print(f"Trained has been saved to: {train_path}")
    print(f"Tested has been saved to: {test_path}")
else:
    raise RuntimeError("Preprocessing error")
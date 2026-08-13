from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder



project_folder = Path(__file__).parent
csv_files = list(project_folder.rglob("*.csv"))

csv_path = csv_files[0]

df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)
print("Target column:", df.columns[-1])



df.columns = df.columns.str.strip()



import numpy as np

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(inplace=True)

print("Shape after cleaning:", df.shape)




X = df.drop("Label", axis=1)
y = df["Label"]



label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("Classes:", label_encoder.classes_)




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)



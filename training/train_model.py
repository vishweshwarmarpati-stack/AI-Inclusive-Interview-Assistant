import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("data/isl_landmarks.csv")

print("Dataset loaded!")
print("Number of samples:", len(data))

# Show signs
print("\nSigns found:")
print(data["label"].value_counts())

# =========================
# INPUT AND OUTPUT
# =========================

X = data.drop("label", axis=1)
y = data["label"]

# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# =========================
# CREATE MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# =========================
# TRAIN
# =========================

print("\nTraining model...")

model.fit(X_train, y_train)

# =========================
# TEST
# =========================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel trained successfully!")
print("Accuracy:", accuracy)

# =========================
# SAVE MODEL
# =========================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/sign_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/sign_model.pkl")
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "isl_landmarks_2hand.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "data", "models", "sign_model_2hand.pkl"
)

print("Loading 2-hand dataset...")

# CSV has NO header
columns = ["label"] + [f"feature_{i}" for i in range(126)]

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=columns
)

print("Dataset shape:", df.shape)

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

print("\nSigns found:")
print(y.value_counts())

# Make sure we have exactly 126 features
if X.shape[1] != 126:
    raise ValueError(
        f"Expected 126 features, but found {X.shape[1]}"
    )

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train model
print("\nTraining 2-hand Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully!")

# Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)

# run_training.py
from src.data_preprocessing import data_preprocess
from src.model_training import train_model

# Step 1: Preprocess
X_train, X_test, y_train, y_test = data_preprocess()

# Step 2: Train and Save
model = train_model(X_train, y_train)

print("✅ Model training complete.")

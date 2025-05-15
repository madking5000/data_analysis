# run_training.py
from src.data_processing import data_preprocess
from src.model_train import train_model
import pandas as pd

# Step 1: Preprocess
data = pd.read_csv('F:/Data science/projects/data_analysis/titanic_project/data/Titanic-Dataset.csv')
X_train, X_test, y_train, y_test = data_preprocess(data)

# Step 2: Train and Save
model = train_model(X_train, y_train)



#lets find the best model and save it locally  
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.ensemble import RandomForestClassifier
import os


#create a function that test different parameters using grid search and then return best model with parameters 

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [10,20,50,100,1000],
        'max_depth': [4, 6, 8]
    }

    # Create models/ directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(model_dir, 'model.pkl')
    
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)
    joblib.dump(grid.best_estimator_, model_path)

    return grid.best_estimator_
# Wine-Quality-Prediction
# Step 1: Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib  # Corrected import
from skimage import io, color, transform
# Step 2: Load and Preprocess Data
def load_data():
    # Load dataset (Assume a dataset with labeled medical images)
    # ...

    # Preprocess images (Assume images are resized to a common size)
    # ...

    return X, y

# Step 3: Split Data into Training and Testing Sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 4: Build and Train the Model
def build_and_train_model(X_train, y_train):
    # Create a pipeline with StandardScaler, PCA, and SVM
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
    ])

    # Train the model
    model.fit(X_train, y_train)
    return model

# Step 5: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Step 6: Save the Model
def save_model(model, filename='image_recognition_model.pkl'):
    joblib.dump(model, filename)

# Step 7: Deployment (Not covered in this simplified example)

# Step 8: Main Function
def main():
    # Step 1: Load Data
    X, y = load_data()

    # Step 2: Split Data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Build and Train Model
    model = build_and_train_model(X_train, y_train)

    # Step 4: Evaluate Model
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')

    # Step 5: Save Model
    save_model(model)

if __name__ == '__main__':
    main()

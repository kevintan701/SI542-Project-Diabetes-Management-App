"""
Data Preprocessing and Model Training Module for Diabetes Risk Assessment

This script implements the data preprocessing pipeline and model training for the 
diabetes risk prediction system. It handles data validation, feature engineering, 
model training, and evaluation.

Features:
- Data Loading and Validation:
  * CSV file input handling
  * Required column validation
  * Missing value detection
  * Data integrity checks

- Data Preprocessing:
  * Feature selection
  * Missing value imputation
  * Non-numeric column removal
  * Feature scaling using StandardScaler

- Model Training:
  * XGBoost Regressor implementation
  * Hyperparameter configuration
  * Cross-validation
  * Train-test split (80-20)

- Model Evaluation:
  * RMSE (Root Mean Square Error)
  * MAE (Mean Absolute Error)
  * R² Score
  * Cross-validation scores
  * Feature importance visualization

- Artifact Management:
  * Model persistence (diabetes_risk_model.pkl)
  * Scaler persistence (scaler.pkl)
  * Feature importance plot (feature_importance.png)

Dependencies:
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Data preprocessing and evaluation
- xgboost: Machine learning model
- matplotlib: Visualization
- joblib: Model persistence
- logging: Error tracking and debugging

Input:
- diabetes_data.csv: Contains patient records with health metrics

Output:
- diabetes_risk_model.pkl: Trained XGBoost model
- scaler.pkl: Fitted StandardScaler
- feature_importance.png: Feature importance visualization
- Logging information: Training metrics and error tracking

Usage:
    python datapreprocessing&modeltrainning.py

Authors: Kevin Tan, Haichao Min, Hanfu Hou, Shreyas Karnad, You Wu, Donald Su
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_validate_data(file_path):
    try:
        data = pd.read_csv(file_path)
        required_columns = [
            'user_id', 'date', 'weight', 'height', 'bmi',
            'blood_glucose', 'physical_activity', 'diet',
            'medication_adherence', 'stress_level', 'sleep_hours',
            'hydration_level', 'risk_score'
        ]

        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Required: {required_columns}")

        logging.info(f"Data loaded successfully with {len(data)} records")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        # Remove non-numeric columns
        data_processed = data.drop(columns=['user_id', 'date'])
        
        # Check for missing values
        missing_values = data_processed.isnull().sum()
        if missing_values.any():
            logging.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")
            data_processed.fillna(data_processed.mean(), inplace=True)
        
        # Add activity level based on physical activity minutes
        def determine_activity_level(minutes):
            if minutes < 30:
                return 0  # low
            elif minutes < 60:
                return 1  # moderate
            else:
                return 2  # high
        
        # Verify feature encoding
        logging.info("Feature encoding verification:")
        logging.info(f"Diet values: {data_processed['diet'].unique()}")  # Should be [0, 1]
        logging.info(f"Medication adherence values: {data_processed['medication_adherence'].unique()}")  # Should be [0, 1]
        logging.info(f"Stress level values: {data_processed['stress_level'].unique()}")  # Should be [0, 1, 2]
        logging.info(f"Hydration level values: {data_processed['hydration_level'].unique()}")  # Should be [0, 1]
        logging.info("Physical activity ranges:")
        logging.info(f"Min: {data_processed['physical_activity'].min():.1f} minutes")
        logging.info(f"Max: {data_processed['physical_activity'].max():.1f} minutes")
        logging.info(f"Mean: {data_processed['physical_activity'].mean():.1f} minutes")
        
        # Split features and target
        X = data_processed.drop(columns=['risk_score'])
        y = data_processed['risk_score']
        
        return X, y
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise

def train_and_evaluate_model(X, y):
    try:
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Log feature names for reference
        feature_names = X.columns.tolist()
        logging.info(f"Features being used: {feature_names}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Model Selection Rationale:
        # We chose XGBoost over other models for several key reasons:
        # 1. Superior Performance: XGBoost consistently outperforms traditional models like
        #    linear regression and random forests for complex healthcare data
        # 2. Feature Importance: Built-in feature importance analysis helps identify key 
        #    health metrics driving risk scores
        # 3. Handling Non-Linear Relationships: Can capture complex non-linear relationships
        #    between health metrics and risk scores
        # 4. Robustness: Handles missing values and outliers well, which is common in
        #    health data
        # 5. Speed: Fast training and prediction times, essential for real-time risk assessment
        # 6. Memory Efficiency: Optimized implementation for large healthcare datasets

        # Initialize and train model with carefully chosen parameters
        model = xgb.XGBRegressor(
            # Use squared error loss since this is a regression problem for continuous risk scores
            objective='reg:squarederror',

            # Set 100 trees as a balanced choice between model complexity and training time
            # More trees could improve accuracy but risk overfitting
            n_estimators=100,

            # Learning rate of 0.1 provides good balance of learning speed and stability
            # Lower values would be more precise but slower, higher values risk overshooting
            learning_rate=0.1,

            # Max depth of 5 prevents overfitting while capturing important feature interactions
            # Deeper trees could model more complex patterns but increase overfitting risk
            max_depth=5,

            # Fixed random seed for reproducible results across training runs
            random_state=42
        )
        
        # Train model
        logging.info("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        
        logging.info(f"Model Performance Metrics:")
        logging.info(f"RMSE: {rmse:.2f}")
        logging.info(f"MAE: {mae:.2f}")
        logging.info(f"R2 Score: {r2:.2f}")
        logging.info(f"Cross-validation scores: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        return model, scaler
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise
def save_artifacts(model, scaler):
    try:
        joblib.dump(model, 'diabetes_risk_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        logging.info("Model and scaler saved successfully")
    except Exception as e:
        logging.error(f"Error saving artifacts: {e}")
        raise

def main():
    try:
        # Load and process data
        data = load_and_validate_data('diabetes_data.csv')
        X, y = preprocess_data(data)
        
        # Train and evaluate model
        model, scaler = train_and_evaluate_model(X, y)
        
        # Save model and scaler
        save_artifacts(model, scaler)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(model, max_num_features=10)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

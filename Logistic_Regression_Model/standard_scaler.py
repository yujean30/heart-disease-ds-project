import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Re-fit scaler on unscaled numerical columns from training data
X_train_raw = pd.read_csv('heart_disease.csv').drop(columns=['Heart Disease Status'])
num_cols = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours', 
            'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level']

scaler = StandardScaler()
scaler.fit(X_train_raw[num_cols].fillna(X_train_raw[num_cols].median()))

# Save scaler inside your baseline folder
os.makedirs('Logistic_Regression_Model', exist_ok=True)
joblib.dump(scaler, 'Logistic_Regression_Model/scaler.pkl')
print("Scaler saved to 'Logistic_Regression_Model/scaler.pkl'")
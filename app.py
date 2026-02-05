from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load artifacts from model folder
MODEL_PATH = "churn_model_artifacts"
model = joblib.load(f'{MODEL_PATH}/churn_prediction_model.pkl')
scaler = joblib.load(f'{MODEL_PATH}/feature_scaler.pkl')
model_features = joblib.load(f'{MODEL_PATH}/model_features.pkl')
categorical_cols = joblib.load(f'{MODEL_PATH}/categorical_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None, css_class=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        customer_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }
        
        # Create DataFrame WITH column names preserved
        df_new = pd.DataFrame([customer_data])
        
        # Handle missing TotalCharges
        if pd.isna(df_new['TotalCharges'].iloc[0]) or df_new['TotalCharges'].iloc[0] == 0:
            df_new['TotalCharges'] = df_new['tenure'].iloc[0] * df_new['MonthlyCharges'].iloc[0]
        
        # Encode categorical columns
        for col in categorical_cols:
            if col in df_new.columns:
                # Use safe encoding that matches training
                le = LabelEncoder()
                # Fit on values that match Telco dataset categories
                dummy_vals = ['No', 'Yes', 'Male', 'Female', 'No phone service', 
                            'DSL', 'Fiber optic', 'No internet service',
                            'Month-to-month', 'One year', 'Two year',
                            'Bank transfer (automatic)', 'Credit card (automatic)',
                            'Electronic check', 'Mailed check']
                le.fit(pd.Series(dummy_vals).astype(str))
                # Transform safely
                df_new[col] = le.transform(df_new[col].astype(str))
        
        # CRITICAL: Reorder columns to EXACTLY match training order
        df_new = df_new[model_features]
        
        # Scale features
        X_scaled = scaler.transform(df_new)
        
        # Predict
        proba = model.predict_proba(X_scaled)[0][1]
        
        # Determine risk level
        if proba >= 0.7:
            risk = "HIGH RISK 🔴 - Immediate intervention needed"
            css_class = "high"
        elif proba >= 0.4:
            risk = "MEDIUM RISK 🟠 - Monitor closely"
            css_class = "medium"
        else:
            risk = "LOW RISK 🟢 - Standard engagement"
            css_class = "low"
        
        return render_template('index.html',
                             prediction=risk,
                             probability=f"{proba*100:.1f}%",
                             css_class=css_class)
                             
    except Exception as e:
        return render_template('index.html',
                             prediction=f"Error: {str(e)}",
                             css_class="error")

if __name__ == '__main__':
    # Disable debug mode to prevent weird Windows reload issues
    app.run(debug=False)
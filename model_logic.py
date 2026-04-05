import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
import shap

def get_model_insights(df, model, encoders, metadata):
    # 1. Preprocess
    df_proc = df.copy()
    for col, le in encoders.items():
        df_proc[col] = df_proc[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
        df_proc[col] = le.transform(df_proc[col])
    
    X = df_proc.drop(['transaction id', 'timestamp', 'fraud_flag', 'transaction_status'], axis=1, errors='ignore')
    
    # 2. Get Trust Score (Soft-Maximum Logic)
    probs = model.predict_proba(X)[:, 1]
    alpha = 5
    soft_max_risk = np.log(np.sum(np.exp(alpha * probs))) / alpha
    trust_score = max(0, min(100, (1 - soft_max_risk) * 100))
    
    # 3. SHAP Explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Sum SHAP values across all 50 transactions to find overall drivers
    # Note: For binary XGBoost, shap_values is often a single array for Class 1
    mean_shap = np.mean(shap_values, axis=0) 
    feature_names = X.columns
    
    # Create a list of (Feature, Impact) sorted by absolute impact
    insights = sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)
    
    # 4. Reason Mapping Template
    templates = {
        'amount': ("High transaction volumes detected", "Consistent, low-risk spending patterns"),
        'transaction type': ("Atypical transaction methods used", "Standard payment channels maintained"),
        'device_type': ("Multiple unrecognized devices used", "Secure, recognized device history"),
        'merchant_category': ("Spending in high-risk categories", "Spending in stable retail categories"),
        'sender_bank': ("Frequent transfers from volatile accounts", "Stable banking relationships"),
        'network_type': ("Transactions over insecure networks", "Secure network usage detected")
    }
    
    reasons = []
    # If Trust Score is Low (< 60), get top 5 Negative impacts (Highest SHAP for fraud)
    if trust_score < 80:
        negative_drivers = sorted([i for i in insights if i[1] > 0], key=lambda x: x[1], reverse=True)[:5]
        for feat, val in negative_drivers:
            msg = templates.get(feat, (f"Anomaly in {feat}", ""))[0]
            reasons.append({"feature": feat, "text": msg, "type": "negative"})
    else:
        # If Trust Score is High, get top 5 Positive impacts (Lowest SHAP/Negative SHAP)
        positive_drivers = sorted([i for i in insights if i[1] < 0], key=lambda x: x[1])[:5]
        for feat, val in positive_drivers:
            msg = templates.get(feat, ("", f"Verified stability in {feat}"))[1]
            reasons.append({"feature": feat, "text": msg, "type": "positive"})
            
    return trust_score, reasons
"""
Production Inference & Advisory Engine
Combines champion ML models, financial analytics, and explainable decision factors.
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

class RealEstateAdvisorPredictor:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.preprocessor = joblib.load(self.models_dir / "preprocessor.joblib")
        self.feature_engineer = joblib.load(self.models_dir / "feature_engineer.joblib")
        self.classifier = joblib.load(self.models_dir / "champion_classifier.joblib")
        self.regressor = joblib.load(self.models_dir / "champion_regressor.joblib")
        
        # Load feature columns
        import json
        with open(self.models_dir / "model_leaderboard.json", "r") as f:
            metadata = json.load(f)
            self.cat_cols = metadata["feature_columns"]["categorical"]
            self.num_cols = metadata["feature_columns"]["numeric"]

    def predict(self, property_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts a dictionary of property attributes and returns end-to-end investment evaluation.
        """
        df_input = pd.DataFrame([property_dict])
        
        # 1. Preprocessing
        df_clean = self.preprocessor.transform(df_input, handle_outliers=False)
        
        # 2. Feature Engineering
        df_feat = self.feature_engineer.transform(df_clean)
        
        # Extract features for model inference
        X_features = df_feat[self.cat_cols + self.num_cols]
        
        # 3. Model Predictions
        # Future Price (Regression)
        pred_future_price_5y = float(self.regressor.predict(X_features)[0])
        pred_future_price_5y = max(pred_future_price_5y, float(df_feat["Price_in_Lakhs"].iloc[0]))
        
        # Good Investment (Classification)
        if hasattr(self.classifier, "predict_proba"):
            proba_arr = self.classifier.predict_proba(X_features)[0]
            good_inv_prob = float(proba_arr[1])
        else:
            pred_class = int(self.classifier.predict(X_features)[0])
            good_inv_prob = 0.85 if pred_class == 1 else 0.20
            
        is_good_investment = good_inv_prob >= 0.50
        
        # 4. Financial Analytics
        current_price = float(df_feat["Price_in_Lakhs"].iloc[0])
        size_sqft = float(df_feat["Size_in_SqFt"].iloc[0])
        psf = float(df_feat["Price_per_SqFt"].iloc[0])
        
        net_gain_5y = round(pred_future_price_5y - current_price, 2)
        roi_5y = round((net_gain_5y / current_price) * 100.0, 2)
        cagr = round(((pred_future_price_5y / current_price) ** (1/5) - 1) * 100.0, 2)
        
        # Confidence interval (+/- 5.5% model error band)
        ci_lower = round(pred_future_price_5y * 0.945, 2)
        ci_upper = round(pred_future_price_5y * 1.055, 2)
        
        # Trajectory for years 1 through 5
        annual_rate = (pred_future_price_5y / current_price) ** (1/5) - 1
        trajectory = []
        for year in range(0, 6):
            val = round(current_price * ((1 + annual_rate) ** year), 2)
            trajectory.append({
                "Year": f"Year {year}" if year > 0 else "Present",
                "Estimated_Price_Lakhs": val
            })
            
        # Rental Analytics (Typical Indian metro rental yield: 2.8% to 3.8%)
        rental_yield_pct = 3.2
        if df_feat["Public_Transport_Accessibility"].iloc[0] == "High":
            rental_yield_pct += 0.4
        if df_feat["Furnished_Status"].iloc[0] == "Fully Furnished":
            rental_yield_pct += 0.5
            
        annual_rental_lakhs = (current_price * rental_yield_pct) / 100.0
        monthly_rental_inr = round((annual_rental_lakhs * 100000) / 12.0)
        
        # EMI Calculation (80% loan, 8.5% interest, 20-year tenure)
        loan_amount = current_price * 0.80 * 100000 # in INR
        monthly_rate = 8.5 / (12 * 100)
        tenure_months = 20 * 12
        emi = round((loan_amount * monthly_rate * ((1 + monthly_rate) ** tenure_months)) / (((1 + monthly_rate) ** tenure_months) - 1))
        
        # Investment Rating Grade
        if good_inv_prob >= 0.82:
            rating_grade = "AAA (Prime High-Growth)"
            risk_level = "Low Risk"
        elif good_inv_prob >= 0.65:
            rating_grade = "AA (Solid Investment)"
            risk_level = "Moderate-Low Risk"
        elif good_inv_prob >= 0.50:
            rating_grade = "A (Balanced / Hold)"
            risk_level = "Moderate Risk"
        elif good_inv_prob >= 0.35:
            rating_grade = "BBB (Speculative)"
            risk_level = "Elevated Risk"
        else:
            rating_grade = "C (High Risk / Overvalued)"
            risk_level = "High Risk"
            
        # Decision factors & Drivers
        locality_median = float(df_feat["Locality_Median_PSF"].iloc[0])
        psf_ratio = float(df_feat["Locality_PSF_Ratio"].iloc[0])
        infra_score = float(df_feat["Infrastructure_Score"].iloc[0])
        
        decision_factors = []
        if psf_ratio < 0.95:
            discount_pct = round((1 - psf_ratio) * 100, 1)
            decision_factors.append(f"✅ Bargain Pricing: Priced {discount_pct}% below locality median PSF (₹{locality_median:,.0f}/sqft).")
        elif psf_ratio > 1.15:
            premium_pct = round((psf_ratio - 1) * 100, 1)
            decision_factors.append(f"⚠️ Premium Surcharge: Trading at {premium_pct}% premium to locality median PSF.")
        else:
            decision_factors.append(f"ℹ️ Fair Valuation: Aligned with current locality benchmarks.")
            
        if infra_score >= 70:
            decision_factors.append(f"✅ High Infrastructure Index ({infra_score}/100): Excellent schools, hospitals, and transit connectivity.")
        elif infra_score < 45:
            decision_factors.append(f"⚠️ Developing Infrastructure ({infra_score}/100): Limited nearby healthcare/schools.")
            
        if cagr >= 9.5:
            decision_factors.append(f"✅ Strong Capital Appreciation: Forecasted CAGR of {cagr}% exceeds inflation and equity hurdle rates.")
        else:
            decision_factors.append(f"ℹ️ Moderate Growth: Forecasted CAGR of {cagr}%.")
            
        return {
            "current_price_lakhs": round(current_price, 2),
            "predicted_future_price_5y": round(pred_future_price_5y, 2),
            "ci_lower_5y": ci_lower,
            "ci_upper_5y": ci_upper,
            "net_gain_5y_lakhs": net_gain_5y,
            "roi_5y_percent": roi_5y,
            "annualized_cagr_percent": cagr,
            "good_investment": is_good_investment,
            "confidence_score": round(good_inv_prob * 100.0, 1),
            "rating_grade": rating_grade,
            "risk_level": risk_level,
            "rental_yield_percent": round(rental_yield_pct, 2),
            "estimated_monthly_rent_inr": monthly_rental_inr,
            "estimated_monthly_emi_inr": emi,
            "infrastructure_score": round(infra_score, 1),
            "price_per_sqft": round(psf, 2),
            "locality_median_psf": round(locality_median, 2),
            "psf_ratio": round(psf_ratio, 2),
            "trajectory": trajectory,
            "decision_factors": decision_factors
        }

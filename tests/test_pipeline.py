"""
Automated Pipeline and Model Verification Tests
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import HousingDataPreprocessor
from src.feature_engineering import HousingFeatureEngineer
from src.predictor import RealEstateAdvisorPredictor

def test_preprocessing():
    sample_df = pd.DataFrame([{
        "ID": "PROP_9999",
        "State": "Maharashtra",
        "City": "Mumbai",
        "Locality": "Bandra West",
        "Property_Type": "Apartment",
        "BHK": 3,
        "Size_in_SqFt": 1500,
        "Price_in_Lakhs": 350.0,
        "Price_per_SqFt": 23333.33,
        "Year_Built": 2020,
        "Furnished_Status": np.nan,
        "Floor_No": 5,
        "Total_Floors": 15,
        "Nearby_Schools": 8,
        "Nearby_Hospitals": 4,
        "Public_Transport_Accessibility": "High",
        "Parking_Space": np.nan,
        "Security": "24x7 Security & CCTV",
        "Amenities": "Gym, Pool",
        "Facing": "East",
        "Owner_Type": "Individual",
        "Availability_Status": "Ready to Move"
    }])
    
    preprocessor = HousingDataPreprocessor().fit(sample_df)
    clean_df = preprocessor.transform(sample_df)
    
    assert clean_df["Furnished_Status"].isna().sum() == 0, "Missing Furnished_Status should be imputed."
    assert clean_df["Parking_Space"].isna().sum() == 0, "Missing Parking_Space should be imputed."
    assert "Age_of_Property" in clean_df.columns, "Age_of_Property should be calculated."
    assert clean_df["Age_of_Property"].iloc[0] == (2026 - 2020)

def test_feature_engineering():
    sample_df = pd.DataFrame([{
        "ID": "PROP_9999",
        "State": "Karnataka",
        "City": "Bengaluru",
        "Locality": "Indiranagar",
        "Property_Type": "Apartment",
        "BHK": 3,
        "Size_in_SqFt": 1600,
        "Price_in_Lakhs": 220.0,
        "Price_per_SqFt": 13750.0,
        "Year_Built": 2022,
        "Age_of_Property": 4,
        "Furnished_Status": "Fully Furnished",
        "Floor_No": 4,
        "Total_Floors": 12,
        "Nearby_Schools": 9,
        "Nearby_Hospitals": 6,
        "Public_Transport_Accessibility": "High",
        "Parking_Space": 2,
        "Security": "24x7 Security & CCTV",
        "Amenities": "Gym, Swimming Pool, Clubhouse",
        "Facing": "North-East",
        "Owner_Type": "Individual",
        "Availability_Status": "Ready to Move"
    }])
    
    engineer = HousingFeatureEngineer().fit(sample_df)
    feat_df = engineer.transform(sample_df)
    
    assert "Infrastructure_Score" in feat_df.columns
    assert "Transport_Score" in feat_df.columns
    assert "Amenity_Count" in feat_df.columns
    assert feat_df["Infrastructure_Score"].iloc[0] > 0
    assert feat_df["Amenity_Count"].iloc[0] == 3

def test_predictor_end_to_end():
    predictor = RealEstateAdvisorPredictor()
    sample = {
        "State": "Maharashtra",
        "City": "Mumbai",
        "Locality": "Andheri West",
        "Property_Type": "Apartment",
        "BHK": 3,
        "Size_in_SqFt": 1400,
        "Price_in_Lakhs": 280.0,
        "Year_Built": 2022,
        "Furnished_Status": "Fully Furnished",
        "Floor_No": 8,
        "Total_Floors": 20,
        "Nearby_Schools": 7,
        "Nearby_Hospitals": 4,
        "Public_Transport_Accessibility": "High",
        "Parking_Space": 2,
        "Security": "24x7 Security & CCTV",
        "Amenities": "Gym, Swimming Pool, Clubhouse",
        "Facing": "East",
        "Owner_Type": "Individual",
        "Availability_Status": "Ready to Move"
    }
    
    res = predictor.predict(sample)
    
    assert "predicted_future_price_5y" in res
    assert res["predicted_future_price_5y"] >= res["current_price_lakhs"]
    assert "good_investment" in res
    assert "roi_5y_percent" in res
    assert "trajectory" in res
    assert len(res["trajectory"]) == 6 # Present + 5 Years
    assert "decision_factors" in res
    assert len(res["decision_factors"]) > 0

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_pipeline.py"])


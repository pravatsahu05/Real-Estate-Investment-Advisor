"""
Preprocessing Module for Real Estate Investment Advisor
Handles data cleaning, missing value imputation, outlier detection, and validation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

class HousingDataPreprocessor:
    def __init__(self):
        self.numeric_impute_values: Dict[str, float] = {}
        self.categorical_impute_values: Dict[str, str] = {}
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> "HousingDataPreprocessor":
        """Compute imputation values and outlier bounds from training data."""
        # Clean duplicates first
        df_clean = df.drop_duplicates().copy()

        # Numeric columns to impute
        numeric_cols = [
            "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", 
            "Floor_No", "Total_Floors", "Age_of_Property", 
            "Nearby_Schools", "Nearby_Hospitals", "Parking_Space"
        ]
        default_numeric_fallbacks = {
            "Size_in_SqFt": 1200.0, "Price_in_Lakhs": 100.0, "Price_per_SqFt": 8000.0,
            "Floor_No": 3.0, "Total_Floors": 10.0, "Age_of_Property": 5.0,
            "Nearby_Schools": 5.0, "Nearby_Hospitals": 3.0, "Parking_Space": 1.0
        }
        for col in numeric_cols:
            if col in df_clean.columns:
                med = df_clean[col].median()
                self.numeric_impute_values[col] = float(med) if pd.notna(med) else default_numeric_fallbacks.get(col, 1.0)
            else:
                self.numeric_impute_values[col] = default_numeric_fallbacks.get(col, 1.0)

        # Categorical columns to impute
        categorical_cols = [
            "State", "City", "Locality", "Property_Type", 
            "Furnished_Status", "Public_Transport_Accessibility", 
            "Security", "Amenities", "Facing", "Owner_Type", "Availability_Status"
        ]
        for col in categorical_cols:
            if col in df_clean.columns:
                mode_val = df_clean[col].mode()
                self.categorical_impute_values[col] = str(mode_val[0]) if not mode_val.empty else "Unknown"

        # Outlier bounds for key numeric columns using 1.5 * IQR
        for col in ["Size_in_SqFt", "Price_per_SqFt"]:
            if col in df_clean.columns:
                q25 = df_clean[col].quantile(0.01)
                q75 = df_clean[col].quantile(0.99)
                self.outlier_bounds[col] = (float(q25), float(q75))

        return self

    def transform(self, df: pd.DataFrame, handle_outliers: bool = True) -> pd.DataFrame:
        """Apply imputation, deduplication, and outlier clipping."""
        df_out = df.copy()

        # 1. Drop duplicates
        df_out = df_out.drop_duplicates(subset=["ID"] if "ID" in df_out.columns else None)

        # Compute Age_of_Property if Year_Built is present
        if "Year_Built" in df_out.columns and "Age_of_Property" not in df_out.columns:
            df_out["Age_of_Property"] = 2026 - df_out["Year_Built"]
        elif "Age_of_Property" in df_out.columns and "Year_Built" not in df_out.columns:
            df_out["Year_Built"] = 2026 - df_out["Age_of_Property"]

        # 2. Impute missing values
        for col, val in self.numeric_impute_values.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(val)

        for col, val in self.categorical_impute_values.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(val)

        # 3. Floor number sanity check: Floor_No cannot exceed Total_Floors
        if "Floor_No" in df_out.columns and "Total_Floors" in df_out.columns:
            df_out["Total_Floors"] = np.maximum(df_out["Total_Floors"], df_out["Floor_No"])

        # 4. Outlier clipping (to ensure stability in training/evaluation)
        if handle_outliers:
            for col, (lower, upper) in self.outlier_bounds.items():
                if col in df_out.columns:
                    df_out[col] = df_out[col].clip(lower=lower, upper=upper)

        # 5. Recompute Price_per_SqFt consistency if both Price and Size exist
        if "Price_in_Lakhs" in df_out.columns and "Size_in_SqFt" in df_out.columns:
            df_out["Price_per_SqFt"] = np.round((df_out["Price_in_Lakhs"] * 100000.0) / df_out["Size_in_SqFt"], 2)

        return df_out

    def fit_transform(self, df: pd.DataFrame, handle_outliers: bool = True) -> pd.DataFrame:
        return self.fit(df).transform(df, handle_outliers=handle_outliers)


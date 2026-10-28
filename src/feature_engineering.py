"""
Feature Engineering Module for Real Estate Investment Advisor
Engineers domain-specific financial, infrastructure, and geospatial real estate metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

class HousingFeatureEngineer:
    def __init__(self):
        self.locality_median_psf: Dict[str, float] = {}
        self.city_median_price: Dict[str, float] = {}
        self.global_median_psf: float = 8000.0
        self.global_median_price: float = 120.0

    def fit(self, df: pd.DataFrame) -> "HousingFeatureEngineer":
        """Compute reference group medians from the training dataset."""
        if "Locality" in df.columns and "Price_per_SqFt" in df.columns:
            self.locality_median_psf = df.groupby("Locality")["Price_per_SqFt"].median().to_dict()
            self.global_median_psf = float(df["Price_per_SqFt"].median())

        if "City" in df.columns and "Price_in_Lakhs" in df.columns:
            self.city_median_price = df.groupby("City")["Price_in_Lakhs"].median().to_dict()
            self.global_median_price = float(df["Price_in_Lakhs"].median())

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new domain features according to real estate investment rules."""
        df_feat = df.copy()

        # 1. Price per Sq Ft
        if "Price_in_Lakhs" in df_feat.columns and "Size_in_SqFt" in df_feat.columns:
            df_feat["Price_per_SqFt"] = np.round((df_feat["Price_in_Lakhs"] * 100000.0) / np.maximum(df_feat["Size_in_SqFt"], 1), 2)

        # 2. BHK Size Ratio (Area per BHK)
        if "Size_in_SqFt" in df_feat.columns and "BHK" in df_feat.columns:
            df_feat["BHK_Size_Ratio"] = np.round(df_feat["Size_in_SqFt"] / np.maximum(df_feat["BHK"], 1), 2)

        # 3. Floor Ratio
        if "Floor_No" in df_feat.columns and "Total_Floors" in df_feat.columns:
            df_feat["Floor_Ratio"] = np.round(df_feat["Floor_No"] / np.maximum(df_feat["Total_Floors"], 1), 3)

        # 4. Public Transport Score
        transport_map = {"High": 3.0, "Medium": 2.0, "Low": 1.0}
        if "Public_Transport_Accessibility" in df_feat.columns:
            df_feat["Transport_Score"] = df_feat["Public_Transport_Accessibility"].map(transport_map).fillna(2.0)
        else:
            df_feat["Transport_Score"] = 2.0

        # 5. Amenity Count
        if "Amenities" in df_feat.columns:
            df_feat["Amenity_Count"] = df_feat["Amenities"].astype(str).apply(
                lambda x: len([a.strip() for a in x.split(",") if a.strip()])
            )
        else:
            df_feat["Amenity_Count"] = 2

        # 6. Infrastructure Score (Composite Index: Schools, Hospitals, Transport, Security)
        # Scaled between 0 and 100
        schools = df_feat["Nearby_Schools"] if "Nearby_Schools" in df_feat.columns else 5
        hospitals = df_feat["Nearby_Hospitals"] if "Nearby_Hospitals" in df_feat.columns else 3
        transport = df_feat["Transport_Score"]
        
        # Security score
        def score_security(val):
            val = str(val).lower()
            if "24x7" in val or "cctv" in val and "gated" in val:
                return 10.0
            elif "gated" in val or "guard" in val:
                return 7.5
            elif "cctv" in val:
                return 5.0
            return 3.0

        if "Security" in df_feat.columns:
            sec_score = df_feat["Security"].apply(score_security)
        else:
            sec_score = 6.0

        # Weighted Infrastructure Score
        infra_score = (
            (np.clip(schools / 15.0, 0, 1) * 30.0) +
            (np.clip(hospitals / 10.0, 0, 1) * 25.0) +
            ((transport / 3.0) * 25.0) +
            ((sec_score / 10.0) * 20.0)
        )
        df_feat["Infrastructure_Score"] = np.round(infra_score, 2)

        # 7. School & Hospital Density Scores
        if "Nearby_Schools" in df_feat.columns:
            df_feat["School_Density_Score"] = np.round(np.clip(df_feat["Nearby_Schools"] / 1.5, 0, 10), 2)
        if "Nearby_Hospitals" in df_feat.columns:
            df_feat["Hospital_Density_Score"] = np.round(np.clip(df_feat["Nearby_Hospitals"], 0, 10), 2)

        # 8. Locality Relative Price Index
        # Locality_PSF_Ratio = Property_PSF / Locality_Median_PSF
        # Values < 1.0 indicate an underpriced/bargain property!
        if "Locality" in df_feat.columns and "Price_per_SqFt" in df_feat.columns:
            loc_medians = df_feat["Locality"].map(self.locality_median_psf).fillna(self.global_median_psf)
            df_feat["Locality_Median_PSF"] = loc_medians
            df_feat["Locality_PSF_Ratio"] = np.round(df_feat["Price_per_SqFt"] / np.maximum(loc_medians, 1.0), 3)

        # 9. City Relative Price Index
        if "City" in df_feat.columns and "Price_in_Lakhs" in df_feat.columns:
            city_medians = df_feat["City"].map(self.city_median_price).fillna(self.global_median_price)
            df_feat["City_Median_Price"] = city_medians
            df_feat["City_Price_Ratio"] = np.round(df_feat["Price_in_Lakhs"] / np.maximum(city_medians, 1.0), 3)

        # 10. Facing Vastu Rating (Important domain factor in Indian Real Estate)
        vastu_map = {
            "North": 1.10, "East": 1.10, "North-East": 1.12,
            "North-West": 1.00, "West": 1.00, "South-East": 0.96,
            "South": 0.94, "South-West": 0.92
        }
        if "Facing" in df_feat.columns:
            df_feat["Facing_Vastu_Score"] = df_feat["Facing"].map(vastu_map).fillna(1.0)
        else:
            df_feat["Facing_Vastu_Score"] = 1.0

        # 11. Property Age Category
        if "Age_of_Property" in df_feat.columns:
            df_feat["Age_Category"] = pd.cut(
                df_feat["Age_of_Property"],
                bins=[-1, 3, 10, 20, 100],
                labels=["Brand New (0-3 yrs)", "Recent (4-10 yrs)", "Established (11-20 yrs)", "Legacy (>20 yrs)"]
            ).astype(str)

        return df_feat

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


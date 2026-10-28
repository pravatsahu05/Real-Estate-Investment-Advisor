"""
Model Training and MLflow Tracking Module
Trains Classification & Regression models, logs experiments to MLflow, and exports production pipelines.
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score,
    roc_curve
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
import mlflow
import mlflow.sklearn

from src.preprocessing import HousingDataPreprocessor
from src.feature_engineering import HousingFeatureEngineer

def prepare_pipeline_and_data(csv_path: str = "data/india_housing_prices.csv"):
    raw_df = pd.read_csv(csv_path)
    
    # Preprocess
    preprocessor = HousingDataPreprocessor()
    clean_df = preprocessor.fit_transform(raw_df)
    
    # Feature Engineer
    engineer = HousingFeatureEngineer()
    processed_df = engineer.fit_transform(clean_df)
    
    # Save processed dataframe
    Path("data").mkdir(exist_ok=True)
    processed_df.to_csv("data/processed_housing_data.csv", index=False)
    
    # Save preprocessor & engineer artifacts
    Path("models").mkdir(exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    joblib.dump(engineer, "models/feature_engineer.joblib")
    
    # Feature selection
    categorical_features = [
        "State", "City", "Locality", "Property_Type", 
        "Furnished_Status", "Public_Transport_Accessibility", 
        "Facing", "Owner_Type", "Availability_Status"
    ]
    numeric_features = [
        "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", 
        "Floor_No", "Total_Floors", "Age_of_Property", 
        "Nearby_Schools", "Nearby_Hospitals", "Parking_Space",
        "Transport_Score", "Amenity_Count", "Infrastructure_Score", 
        "BHK_Size_Ratio", "Floor_Ratio", "Locality_PSF_Ratio", 
        "City_Price_Ratio", "Facing_Vastu_Score"
    ]
    
    X = processed_df[categorical_features + numeric_features]
    y_clf = processed_df["Good_Investment"]
    y_reg = processed_df["Future_Price_5Y"]
    
    # Data Split
    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X, y_clf, y_reg, test_size=0.20, random_state=42, stratify=y_clf
    )
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_clf_train": y_clf_train,
        "y_clf_test": y_clf_test,
        "y_reg_train": y_reg_train,
        "y_reg_test": y_reg_test,
        "cat_cols": categorical_features,
        "num_cols": numeric_features,
        "processed_df": processed_df
    }

def train_and_track_models(data_dict: Dict[str, Any]):
    # Setup MLflow sqlite database tracking
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    db_path = Path("mlflow.db").absolute()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Real_Estate_Investment_Advisor")
    
    cat_cols = data_dict["cat_cols"]
    num_cols = data_dict["num_cols"]
    X_train = data_dict["X_train"]
    X_test = data_dict["X_test"]
    y_clf_train = data_dict["y_clf_train"]
    y_clf_test = data_dict["y_clf_test"]
    y_reg_train = data_dict["y_reg_train"]
    y_reg_test = data_dict["y_reg_test"]
    
    artifacts_dir = Path("reports/models")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Define ColumnTransformers
    # One for linear/dense models (with standard scaling)
    preprocessor_dense = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )
    
    # Classification Models
    clf_models = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "preprocessor": preprocessor_dense,
            "params": {"model_type": "LogisticRegression", "C": 1.0, "max_iter": 1000}
        },
        "Random_Forest_Classifier": {
            "model": RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_split=4, random_state=42, n_jobs=-1),
            "preprocessor": preprocessor_dense,
            "params": {"model_type": "RandomForestClassifier", "n_estimators": 150, "max_depth": 16}
        },
        "Hist_Gradient_Boosting_Classifier": {
            "model": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=8, random_state=42),
            "preprocessor": preprocessor_dense,
            "params": {"model_type": "HistGradientBoostingClassifier", "max_iter": 150, "learning_rate": 0.08}
        }
    }
    
    # Regression Models
    reg_models = {
        "Ridge_Regression": {
            "model": Ridge(alpha=10.0, random_state=42),
            "preprocessor": preprocessor_dense,
            "params": {"model_type": "RidgeRegression", "alpha": 10.0}
        },
        "Random_Forest_Regressor": {
            "model": RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_split=4, random_state=42, n_jobs=-1),
            "preprocessor": preprocessor_dense,
            "params": {"model_type": "RandomForestRegressor", "n_estimators": 150, "max_depth": 18}
        },
        "Hist_Gradient_Boosting_Regressor": {
            "model": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=8, random_state=42),
            "preprocessor": preprocessor_dense,
            "params": {"model_type": "HistGradientBoostingRegressor", "max_iter": 150, "learning_rate": 0.08}
        }
    }
    
    clf_results = {}
    best_clf_f1 = -1.0
    best_clf_pipeline = None
    best_clf_name = ""
    
    print("\n========== Training & Tracking Classification Models ==========")
    for name, item in clf_models.items():
        with mlflow.start_run(run_name=f"CLF_{name}"):
            pipeline = Pipeline([
                ("preprocessor", item["preprocessor"]),
                ("classifier", item["model"])
            ])
            pipeline.fit(X_train, y_clf_train)
            
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
            
            acc = float(accuracy_score(y_clf_test, y_pred))
            prec = float(precision_score(y_clf_test, y_pred, zero_division=0))
            rec = float(recall_score(y_clf_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_clf_test, y_pred, zero_division=0))
            auc = float(roc_auc_score(y_clf_test, y_proba))
            
            mlflow.log_params(item["params"])
            mlflow.log_metrics({
                "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "roc_auc": auc
            })
            
            # Confusion Matrix Plot
            cm = confusion_matrix(y_clf_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Not Recommended", "Good Investment"],
                        yticklabels=["Not Recommended", "Good Investment"])
            plt.title(f"Confusion Matrix: {name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            cm_path = artifacts_dir / f"cm_{name}.png"
            plt.savefig(cm_path, dpi=160)
            plt.close()
            mlflow.log_artifact(str(cm_path))
            
            # ROC Curve Plot
            fpr, tpr, _ = roc_curve(y_clf_test, y_proba)
            plt.figure(figsize=(5, 4))
            plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {auc:.3f}")
            plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
            plt.title(f"ROC Curve: {name}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_path = artifacts_dir / f"roc_{name}.png"
            plt.savefig(roc_path, dpi=160)
            plt.close()
            mlflow.log_artifact(str(roc_path))
            
            # Log Model Artifact with cloudpickle serialization
            try:
                mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")
            except Exception as e:
                print(f"Notice during MLflow model logging: {e}")
            
            clf_results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "roc_auc": auc}
            print(f"[{name}] Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
            
            if f1 > best_clf_f1:
                best_clf_f1 = f1
                best_clf_pipeline = pipeline
                best_clf_name = name

    # Save champion classifier
    joblib.dump(best_clf_pipeline, "models/champion_classifier.joblib")
    print(f"\nChampion Classifier: {best_clf_name} (F1 = {best_clf_f1:.4f}) saved to models/champion_classifier.joblib")

    # Regression Models
    reg_results = {}
    best_reg_r2 = -999.0
    best_reg_pipeline = None
    best_reg_name = ""
    
    print("\n========== Training & Tracking Regression Models ==========")
    for name, item in reg_models.items():
        with mlflow.start_run(run_name=f"REG_{name}"):
            pipeline = Pipeline([
                ("preprocessor", item["preprocessor"]),
                ("regressor", item["model"])
            ])
            pipeline.fit(X_train, y_reg_train)
            
            y_pred = pipeline.predict(X_test)
            
            rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_pred)))
            mae = float(mean_absolute_error(y_reg_test, y_pred))
            r2 = float(r2_score(y_reg_test, y_pred))
            mape = float(np.mean(np.abs((y_reg_test - y_pred) / y_reg_test)) * 100.0)
            
            mlflow.log_params(item["params"])
            mlflow.log_metrics({
                "rmse": rmse, "mae": mae, "r2_score": r2, "mape_percent": mape
            })
            
            # Residual Plot
            residuals = y_reg_test - y_pred
            plt.figure(figsize=(5, 4))
            sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, color="#059669")
            plt.axhline(0, color="red", linestyle="--")
            plt.title(f"Residuals: {name} (R² = {r2:.3f})")
            plt.xlabel("Predicted Future Price (₹ Lakhs)")
            plt.ylabel("Residuals (₹ Lakhs)")
            plt.tight_layout()
            res_path = artifacts_dir / f"residual_{name}.png"
            plt.savefig(res_path, dpi=160)
            plt.close()
            mlflow.log_artifact(str(res_path))
            
            # Log Model Artifact with cloudpickle serialization
            try:
                mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")
            except Exception as e:
                print(f"Notice during MLflow model logging: {e}")
            
            reg_results[name] = {"rmse": rmse, "mae": mae, "r2_score": r2, "mape_percent": mape}
            print(f"[{name}] R2: {r2:.4f} | RMSE: INR {rmse:.2f}L | MAE: INR {mae:.2f}L | MAPE: {mape:.2f}%")
            
            if r2 > best_reg_r2:
                best_reg_r2 = r2
                best_reg_pipeline = pipeline
                best_reg_name = name

    # Save champion regressor
    joblib.dump(best_reg_pipeline, "models/champion_regressor.joblib")
    print(f"\nChampion Regressor: {best_reg_name} (R2 = {best_reg_r2:.4f}) saved to models/champion_regressor.joblib")
    
    # Save Model Comparison Leaderboard Metadata
    leaderboard = {
        "classification": clf_results,
        "regression": reg_results,
        "champion_classifier": best_clf_name,
        "champion_regressor": best_reg_name,
        "feature_columns": {
            "categorical": cat_cols,
            "numeric": num_cols
        }
    }
    with open("models/model_leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)
        
    print("\nTraining and tracking successfully completed! All models and artifacts registered.")
    return leaderboard

if __name__ == "__main__":
    data_dict = prepare_pipeline_and_data()
    train_and_track_models(data_dict)

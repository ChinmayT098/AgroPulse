import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data', 'agricultural_dataset.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'crop_yield_rf_model.pkl')
METRICS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'model_metrics.pkl')

CATEGORICAL_FEATURES = ['Crop', 'Season', 'State', 'Soil_Type']
NUMERICAL_FEATURES = [
    'Area_Hectares',
    'Rainfall_mm',
    'Temperature_C',
    'Soil_pH',
    'Nitrogen_N',
    'Phosphorus_P',
    'Potassium_K'
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COLUMN = 'Yield_ton_per_ha'


class CropYieldModelManager:
    """Manages training, evaluating, and predicting crop yields using Random Forest Regressor."""

    def __init__(self, data_path=DATASET_PATH):
        self.data_path = data_path
        self.pipeline = None
        self.metrics = None
        self.feature_importances = None
        self.crop_stats = {}
        self.df = None
        self._load_data()
        self._initialize_or_train()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        self.df = pd.read_csv(self.data_path)

        # Precompute crop statistics for context and benchmarking
        grouped = self.df.groupby('Crop')['Yield_ton_per_ha']
        self.crop_stats = {
            crop: {
                'avg_yield': round(float(grouped.get_group(crop).mean()), 2),
                'min_yield': round(float(grouped.get_group(crop).min()), 2),
                'max_yield': round(float(grouped.get_group(crop).max()), 2),
                'count': int(grouped.get_group(crop).count())
            }
            for crop in grouped.groups.keys()
        }

    def _initialize_or_train(self):
        """Train genuine Random Forest model and compute evaluation metrics."""
        self.train_and_evaluate()

    def train_and_evaluate(self):
        """Trains the Random Forest Regression model and records real evaluation metrics."""
        X = self.df[ALL_FEATURES]
        y = self.df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
                ('num', 'passthrough', NUMERICAL_FEATURES)
            ]
        )

        rf = RandomForestRegressor(
            n_estimators=120,
            max_depth=16,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', rf)
        ])

        self.pipeline.fit(X_train, y_train)

        # Evaluation on test set
        y_pred = self.pipeline.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        self.metrics = {
            'mae': round(mae, 4),
            'mse': round(mse, 4),
            'rmse': round(rmse, 4),
            'r2_score': round(r2, 4),
            'r2_percentage': round(r2 * 100, 2),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(self.df)
        }

        # Calculate feature importances
        try:
            ohe = self.pipeline.named_steps['preprocessor'].named_transformers_['cat']
            cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
            all_feature_names = cat_feature_names + NUMERICAL_FEATURES
            rf_importances = self.pipeline.named_steps['regressor'].feature_importances_

            importance_df = pd.DataFrame({
                'feature': all_feature_names,
                'importance': rf_importances
            }).sort_values('importance', ascending=False)

            # Summarize categorical groupings for display
            summary_importances = {}
            for col in ALL_FEATURES:
                matching_mask = importance_df['feature'].apply(lambda x: x == col or x.startswith(col + '_'))
                summary_importances[col] = float(importance_df[matching_mask]['importance'].sum())

            # Sort descending
            self.feature_importances = dict(sorted(summary_importances.items(), key=lambda item: item[1], reverse=True))
        except Exception as e:
            print("Feature importance calculation note:", e)
            self.feature_importances = {}

        return self.metrics

    def predict(self, input_data: dict) -> dict:
        """
        Predict crop yield given user input parameters.
        Returns predicted yield, unit, benchmark comparisons, and model confidence metrics.
        """
        # Validate required columns
        for col in ALL_FEATURES:
            if col not in input_data:
                raise ValueError(f"Missing required input feature: {col}")

        input_df = pd.DataFrame([input_data])
        
        # Ensure numerical columns are properly cast
        for col in NUMERICAL_FEATURES:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            if input_df[col].isna().any():
                raise ValueError(f"Invalid numerical value provided for feature '{col}'.")

        # Make genuine model prediction
        pred_yield = float(self.pipeline.predict(input_df)[0])
        pred_yield = max(0.05, round(pred_yield, 2))

        crop = str(input_data['Crop'])
        area = float(input_data['Area_Hectares'])
        estimated_production = round(pred_yield * area, 2)

        stats = self.crop_stats.get(crop, {
            'avg_yield': pred_yield,
            'min_yield': pred_yield * 0.5,
            'max_yield': pred_yield * 1.5
        })

        avg_yield = stats['avg_yield']
        diff_pct = round(((pred_yield - avg_yield) / (avg_yield + 1e-6)) * 100, 1)

        if diff_pct > 7.0:
            status = 'Above Historical Average'
            status_class = 'text-success'
        elif diff_pct < -7.0:
            status = 'Below Historical Average'
            status_class = 'text-warning'
        else:
            status = 'Close to Historical Average'
            status_class = 'text-primary'

        return {
            'crop': crop,
            'predicted_yield': pred_yield,
            'unit': 'Tons / Hectare',
            'area_hectares': area,
            'estimated_production': estimated_production,
            'production_unit': 'Tons',
            'historical_avg_yield': avg_yield,
            'comparison_percent': diff_pct,
            'comparison_status': status,
            'status_class': status_class,
            'model_r2': self.metrics.get('r2_score', 0.90),
            'model_mae': self.metrics.get('mae', 0.15)
        }

    def get_metrics(self):
        return self.metrics

    def get_feature_importances(self):
        return self.feature_importances

    def get_crop_list(self):
        return sorted(list(self.crop_stats.keys()))

    def get_state_list(self):
        return sorted(self.df['State'].unique().tolist())

    def get_season_list(self):
        return sorted(self.df['Season'].unique().tolist())

    def get_soil_type_list(self):
        return sorted(self.df['Soil_Type'].unique().tolist())


# Singleton instance for quick access
_manager_instance = None

def get_model_manager():
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = CropYieldModelManager()
    return _manager_instance

if __name__ == '__main__':
    manager = get_model_manager()
    print("Model Metrics:")
    for k, v in manager.get_metrics().items():
        print(f"  {k}: {v}")
    
    test_sample = {
        'Crop': 'Wheat',
        'State': 'Punjab',
        'Season': 'Rabi',
        'Soil_Type': 'Alluvial',
        'Area_Hectares': 1500.0,
        'Rainfall_mm': 550.0,
        'Temperature_C': 20.0,
        'Soil_pH': 6.8,
        'Nitrogen_N': 110.0,
        'Phosphorus_P': 52.0,
        'Potassium_K': 70.0
    }
    pred = manager.predict(test_sample)
    print("\nSample Prediction Result:")
    for k, v in pred.items():
        print(f"  {k}: {v}")

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from model import get_model_manager, ALL_FEATURES, NUMERICAL_FEATURES
from spark_processing import run_spark_pipeline, OUTPUT_SUMMARY_PATH

app = Flask(__name__)
app.config['SECRET_KEY'] = 'agricultural-big-data-secret-key-2026'
# Set to True by build_static.py when pre-rendering pages for Netlify
app.jinja_env.globals['static_export'] = False

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'agricultural_dataset.csv')


def get_dataset():
    if not os.path.exists(DATA_PATH):
        from data_generator import generate_agricultural_dataset
        generate_agricultural_dataset(DATA_PATH)
    return pd.read_csv(DATA_PATH)


@app.route('/')
def dashboard():
    df = get_dataset()
    manager = get_model_manager()
    metrics = manager.get_metrics()

    # Core Dashboard KPI Metrics
    total_records = int(len(df))
    num_crops = int(df['Crop'].nunique())
    avg_yield = round(float(df['Yield_ton_per_ha'].mean()), 2)
    avg_rainfall = round(float(df['Rainfall_mm'].mean()), 1)
    total_production = round(float(df['Production_tons'].sum() / 1000.0), 1)  # in 1,000s of tons

    # 1. Crop-wise Yield Comparison
    crop_yield_df = df.groupby('Crop')['Yield_ton_per_ha'].mean().reset_index()
    crop_yield_df['Yield_ton_per_ha'] = crop_yield_df['Yield_ton_per_ha'].round(2)
    crop_yield_df = crop_yield_df.sort_values(by='Yield_ton_per_ha', ascending=False)
    crop_names = crop_yield_df['Crop'].tolist()
    crop_yields = crop_yield_df['Yield_ton_per_ha'].tolist()

    # 2. Rainfall vs Crop Yield (Sample 150 points for responsive scatter visualization)
    sample_df = df.sample(n=min(150, len(df)), random_state=42)
    rainfall_scatter = [
        {'x': round(float(row['Rainfall_mm']), 1), 'y': round(float(row['Yield_ton_per_ha']), 2), 'crop': row['Crop']}
        for _, row in sample_df.iterrows()
    ]

    # 3. Historical Productivity Trends (Year vs Avg Yield)
    yearly_df = df.groupby('Year')['Yield_ton_per_ha'].mean().reset_index().sort_values(by='Year')
    yearly_df['Yield_ton_per_ha'] = yearly_df['Yield_ton_per_ha'].round(2)
    trend_years = [str(y) for y in yearly_df['Year'].tolist()]
    trend_yields = yearly_df['Yield_ton_per_ha'].tolist()

    # 4. Soil Type Distribution
    soil_dist = df['Soil_Type'].value_counts().to_dict()
    soil_labels = list(soil_dist.keys())
    soil_counts = list(soil_dist.values())

    # 5. Season Distribution
    season_dist = df['Season'].value_counts().to_dict()

    return render_template(
        'dashboard.html',
        total_records=total_records,
        num_crops=num_crops,
        avg_yield=avg_yield,
        avg_rainfall=avg_rainfall,
        total_production=total_production,
        r2_score=metrics.get('r2_score', 0.99),
        mae=metrics.get('mae', 0.54),
        crop_names_json=json.dumps(crop_names),
        crop_yields_json=json.dumps(crop_yields),
        rainfall_scatter_json=json.dumps(rainfall_scatter),
        trend_years_json=json.dumps(trend_years),
        trend_yields_json=json.dumps(trend_yields),
        soil_labels_json=json.dumps(soil_labels),
        soil_counts_json=json.dumps(soil_counts)
    )


@app.route('/analysis')
def analysis():
    df = get_dataset()
    manager = get_model_manager()

    # Descriptive Statistics
    numeric_cols = [
        'Area_Hectares', 'Rainfall_mm', 'Temperature_C', 'Soil_pH',
        'Nitrogen_N', 'Phosphorus_P', 'Potassium_K', 'Yield_ton_per_ha', 'Production_tons'
    ]
    stats_df = df[numeric_cols].describe().round(2).reset_index()
    stats_records = stats_df.to_dict(orient='records')

    # Missing Values
    missing_info = []
    for col in df.columns:
        m_count = int(df[col].isna().sum())
        m_pct = round((m_count / len(df)) * 100, 2)
        missing_info.append({
            'column': col,
            'data_type': str(df[col].dtype),
            'missing_count': m_count,
            'missing_percent': m_pct,
            'status': 'Clean' if m_count == 0 else 'Has Missing'
        })

    # Correlation Matrix (Correlation with Yield)
    corr_matrix = df[numeric_cols].corr().round(3)
    corr_with_yield = corr_matrix['Yield_ton_per_ha'].drop('Yield_ton_per_ha').to_dict()
    
    corr_features = list(corr_with_yield.keys())
    corr_values = list(corr_with_yield.values())

    # Full correlation matrix dictionary for heatmap view
    corr_matrix_dict = corr_matrix.to_dict()

    # Feature Importance from Random Forest Model
    feature_importances = manager.get_feature_importances()
    fi_labels = list(feature_importances.keys()) if feature_importances else []
    fi_values = [round(v * 100, 2) for v in feature_importances.values()] if feature_importances else []

    # Weather Factor Breakdown (Rainfall Bins)
    df['Rainfall_Category'] = pd.cut(
        df['Rainfall_mm'],
        bins=[0, 600, 1100, 2500],
        labels=['Low (< 600mm)', 'Moderate (600-1100mm)', 'High (> 1100mm)']
    )
    rain_effect = df.groupby('Rainfall_Category', observed=False)['Yield_ton_per_ha'].mean().round(2).to_dict()

    # Temperature Bins
    df['Temp_Category'] = pd.cut(
        df['Temperature_C'],
        bins=[0, 20, 28, 50],
        labels=['Cool (< 20°C)', 'Moderate (20-28°C)', 'Warm (> 28°C)']
    )
    temp_effect = df.groupby('Temp_Category', observed=False)['Yield_ton_per_ha'].mean().round(2).to_dict()

    # Table preview (First 25 records)
    table_preview = df.head(25).to_dict(orient='records')

    return render_template(
        'analysis.html',
        stats_records=stats_records,
        missing_info=missing_info,
        corr_features_json=json.dumps(corr_features),
        corr_values_json=json.dumps(corr_values),
        corr_matrix=corr_matrix_dict,
        numeric_cols=numeric_cols,
        fi_labels_json=json.dumps(fi_labels),
        fi_values_json=json.dumps(fi_values),
        rain_effect=rain_effect,
        temp_effect=temp_effect,
        table_preview=table_preview,
        total_rows=len(df),
        total_cols=len(df.columns)
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    manager = get_model_manager()
    metrics = manager.get_metrics()
    
    crops = manager.get_crop_list()
    states = manager.get_state_list()
    seasons = manager.get_season_list()
    soil_types = manager.get_soil_type_list()

    prediction_result = None
    error_message = None
    form_values = {
        'Crop': 'Wheat',
        'State': 'Punjab',
        'Season': 'Rabi',
        'Soil_Type': 'Alluvial',
        'Area_Hectares': '1200',
        'Rainfall_mm': '550',
        'Temperature_C': '19.5',
        'Soil_pH': '6.8',
        'Nitrogen_N': '110',
        'Phosphorus_P': '48',
        'Potassium_K': '72'
    }

    if request.method == 'POST':
        try:
            form_values = {
                'Crop': request.form.get('Crop', 'Wheat'),
                'State': request.form.get('State', 'Punjab'),
                'Season': request.form.get('Season', 'Rabi'),
                'Soil_Type': request.form.get('Soil_Type', 'Alluvial'),
                'Area_Hectares': request.form.get('Area_Hectares', '1000'),
                'Rainfall_mm': request.form.get('Rainfall_mm', '600'),
                'Temperature_C': request.form.get('Temperature_C', '22'),
                'Soil_pH': request.form.get('Soil_pH', '6.8'),
                'Nitrogen_N': request.form.get('Nitrogen_N', '90'),
                'Phosphorus_P': request.form.get('Phosphorus_P', '40'),
                'Potassium_K': request.form.get('Potassium_K', '60')
            }

            # Validation
            numeric_inputs = {}
            for k in NUMERICAL_FEATURES:
                val = float(form_values[k])
                if val < 0:
                    raise ValueError(f"Value for {k.replace('_', ' ')} cannot be negative.")
                numeric_inputs[k] = val

            payload = {
                'Crop': form_values['Crop'],
                'State': form_values['State'],
                'Season': form_values['Season'],
                'Soil_Type': form_values['Soil_Type'],
                **numeric_inputs
            }

            prediction_result = manager.predict(payload)

        except ValueError as ve:
            error_message = str(ve)
        except Exception as e:
            error_message = f"Prediction error: {str(e)}"

    return render_template(
        'predict.html',
        crops=crops,
        states=states,
        seasons=seasons,
        soil_types=soil_types,
        form_values=form_values,
        prediction=prediction_result,
        error_message=error_message,
        metrics=metrics
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    manager = get_model_manager()
    try:
        data = request.get_json(force=True)
        result = manager.predict(data)
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/spark')
def spark_view():
    # Run or load the PySpark ETL results
    spark_data = run_spark_pipeline(DATA_PATH)

    return render_template(
        'spark.html',
        spark_data=spark_data
    )


if __name__ == '__main__':
    print("Starting Flask Agricultural Big Data Application...")
    app.run(host='127.0.0.1', port=5000, debug=True)

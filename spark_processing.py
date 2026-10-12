"""
Agricultural Big Data Processing using PySpark
----------------------------------------------
Demonstrates Big Data ETL pipeline:
1. Loading agricultural dataset with SparkSession / DataFrame API.
2. Data Cleaning: Filtering invalid / outlier values, handling missing fields.
3. Feature Transformations: Computed columns (e.g. Productivity Category, Production).
4. Aggregations: Crop-level summaries, State-level outputs, and Yearly trends.
5. Exporting processed summaries for analytics and dashboards.

Note:
If PySpark or Java is not installed on the host system, the script provides
a fallback simulation engine executing the exact same Big Data transformations
using Pandas, ensuring continuous availability for demonstration and viva.
"""

import os
import json
import time
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'agricultural_dataset.csv')
OUTPUT_SUMMARY_PATH = os.path.join(os.path.dirname(__file__), 'data', 'spark_aggregated_summary.csv')


def run_spark_pipeline(data_path=CSV_PATH):
    """
    Executes the Big Data processing pipeline.
    Attempts genuine PySpark first; falls back gracefully if PySpark/Java JVM is unavailable.
    """
    start_time = time.time()
    result = {
        'engine': 'PySpark',
        'is_fallback': False,
        'status': 'Success',
        'raw_count': 0,
        'cleaned_count': 0,
        'dropped_count': 0,
        'processing_time_sec': 0.0,
        'crop_summary': [],
        'state_summary': [],
        'yearly_summary': [],
        'spark_code_steps': [
            "1. Initialize SparkSession (CropYieldBigDataPipeline)",
            "2. Read CSV with header=True, inferSchema=True into Spark DataFrame",
            "3. Filter records where Area_Hectares > 0 and Rainfall_mm > 0 and Yield_ton_per_ha > 0",
            "4. withColumn transformations: Compute Production_tons and Categorize Yield",
            "5. groupBy('Crop').agg(avg('Yield_ton_per_ha'), sum('Production_tons'), avg('Rainfall_mm'))",
            "6. groupBy('State').agg(avg('Yield_ton_per_ha'), sum('Production_tons'), count('*'))",
            "7. orderBy(desc('Total_Production_Tons')) and save aggregated summary to disk"
        ]
    }

    try:
        # Check if pyspark is available and JVM can start
        import pyspark
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, avg, sum as spark_sum, count as spark_count, round as spark_round, when

        spark = SparkSession.builder \
            .appName("CropYieldBigDataPipeline") \
            .master("local[*]") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()

        df = spark.read.csv(data_path, header=True, inferSchema=True)
        raw_count = df.count()

        # Cleaning: Filter valid positive records
        cleaned_df = df.filter(
            (col("Area_Hectares") > 0) &
            (col("Rainfall_mm") > 0) &
            (col("Temperature_C") > 0) &
            (col("Yield_ton_per_ha") > 0)
        )
        cleaned_count = cleaned_df.count()

        # Transformations: Add Productivity Category
        transformed_df = cleaned_df.withColumn(
            "Yield_Category",
            when(col("Yield_ton_per_ha") > 15.0, "High Biomass Crop")
            .when(col("Yield_ton_per_ha") > 3.0, "High Yield")
            .when(col("Yield_ton_per_ha") > 1.5, "Medium Yield")
            .otherwise("Low Yield")
        )

        # Aggregation 1: Crop-level summary
        crop_agg = transformed_df.groupBy("Crop").agg(
            spark_round(avg("Yield_ton_per_ha"), 2).alias("Avg_Yield_ton_per_ha"),
            spark_round(avg("Rainfall_mm"), 1).alias("Avg_Rainfall_mm"),
            spark_round(avg("Temperature_C"), 1).alias("Avg_Temp_C"),
            spark_round(spark_sum("Production_tons"), 2).alias("Total_Production_Tons"),
            spark_count("*").alias("Total_Records")
        ).orderBy(col("Total_Production_Tons").desc())

        # Aggregation 2: State-level summary
        state_agg = transformed_df.groupBy("State").agg(
            spark_round(avg("Yield_ton_per_ha"), 2).alias("Avg_Yield_ton_per_ha"),
            spark_round(spark_sum("Production_tons"), 2).alias("Total_Production_Tons"),
            spark_count("*").alias("Total_Records")
        ).orderBy(col("Total_Production_Tons").desc())

        # Aggregation 3: Yearly trend
        yearly_agg = transformed_df.groupBy("Year").agg(
            spark_round(avg("Yield_ton_per_ha"), 2).alias("Avg_Yield_ton_per_ha"),
            spark_round(spark_sum("Production_tons"), 2).alias("Total_Production_Tons")
        ).orderBy("Year")

        # Convert aggregations to Python dictionaries for display
        result['crop_summary'] = [row.asDict() for row in crop_agg.collect()]
        result['state_summary'] = [row.asDict() for row in state_agg.collect()]
        result['yearly_summary'] = [row.asDict() for row in yearly_agg.collect()]

        result['raw_count'] = int(raw_count)
        result['cleaned_count'] = int(cleaned_count)
        result['dropped_count'] = int(raw_count - cleaned_count)
        result['engine'] = f"PySpark v{pyspark.__version__} (Spark Local Cluster)"
        result['is_fallback'] = False

        # Save summary
        crop_df = crop_agg.toPandas()
        crop_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)

        spark.stop()

    except Exception as e:
        # Fallback to Pandas execution when PySpark or Java JVM is not configured on Windows
        result['engine'] = 'PySpark Pipeline Simulation (Pandas Engine Fallback)'
        result['is_fallback'] = True
        result['fallback_reason'] = str(e)

        pdf = pd.read_csv(data_path)
        raw_count = len(pdf)

        # Cleaning filter
        cleaned_pdf = pdf[
            (pdf['Area_Hectares'] > 0) &
            (pdf['Rainfall_mm'] > 0) &
            (pdf['Temperature_C'] > 0) &
            (pdf['Yield_ton_per_ha'] > 0)
        ].copy()
        cleaned_count = len(cleaned_pdf)

        # Transformations
        def categorize_yield(y):
            if y > 15.0:
                return 'High Biomass Crop'
            elif y > 3.0:
                return 'High Yield'
            elif y > 1.5:
                return 'Medium Yield'
            return 'Low Yield'

        cleaned_pdf['Yield_Category'] = cleaned_pdf['Yield_ton_per_ha'].apply(categorize_yield)

        # Crop Aggregation
        crop_agg = cleaned_pdf.groupby('Crop').agg(
            Avg_Yield_ton_per_ha=('Yield_ton_per_ha', 'mean'),
            Avg_Rainfall_mm=('Rainfall_mm', 'mean'),
            Avg_Temp_C=('Temperature_C', 'mean'),
            Total_Production_Tons=('Production_tons', 'sum'),
            Total_Records=('Crop', 'count')
        ).reset_index()

        crop_agg['Avg_Yield_ton_per_ha'] = crop_agg['Avg_Yield_ton_per_ha'].round(2)
        crop_agg['Avg_Rainfall_mm'] = crop_agg['Avg_Rainfall_mm'].round(1)
        crop_agg['Avg_Temp_C'] = crop_agg['Avg_Temp_C'].round(1)
        crop_agg['Total_Production_Tons'] = crop_agg['Total_Production_Tons'].round(2)
        crop_agg = crop_agg.sort_values(by='Total_Production_Tons', ascending=False)

        # State Aggregation
        state_agg = cleaned_pdf.groupby('State').agg(
            Avg_Yield_ton_per_ha=('Yield_ton_per_ha', 'mean'),
            Total_Production_Tons=('Production_tons', 'sum'),
            Total_Records=('State', 'count')
        ).reset_index()
        state_agg['Avg_Yield_ton_per_ha'] = state_agg['Avg_Yield_ton_per_ha'].round(2)
        state_agg['Total_Production_Tons'] = state_agg['Total_Production_Tons'].round(2)
        state_agg = state_agg.sort_values(by='Total_Production_Tons', ascending=False)

        # Yearly Trend
        yearly_agg = cleaned_pdf.groupby('Year').agg(
            Avg_Yield_ton_per_ha=('Yield_ton_per_ha', 'mean'),
            Total_Production_Tons=('Production_tons', 'sum')
        ).reset_index()
        yearly_agg['Avg_Yield_ton_per_ha'] = yearly_agg['Avg_Yield_ton_per_ha'].round(2)
        yearly_agg['Total_Production_Tons'] = yearly_agg['Total_Production_Tons'].round(2)
        yearly_agg = yearly_agg.sort_values(by='Year')

        result['crop_summary'] = crop_agg.to_dict(orient='records')
        result['state_summary'] = state_agg.to_dict(orient='records')
        result['yearly_summary'] = yearly_agg.to_dict(orient='records')

        result['raw_count'] = int(raw_count)
        result['cleaned_count'] = int(cleaned_count)
        result['dropped_count'] = int(raw_count - cleaned_count)

        crop_agg.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    duration = round(time.time() - start_time, 3)
    result['processing_time_sec'] = duration
    return result


if __name__ == '__main__':
    print("=" * 60)
    print("Executing Agricultural Big Data PySpark Processing")
    print("=" * 60)
    output = run_spark_pipeline()
    print(f"Processing Engine    : {output['engine']}")
    print(f"Fallback Mode Active : {output['is_fallback']}")
    print(f"Raw Records Loaded   : {output['raw_count']}")
    print(f"Cleaned Valid Records: {output['cleaned_count']}")
    print(f"Processing Duration  : {output['processing_time_sec']}s")
    print("\n--- Crop Aggregation Summary (Top 5) ---")
    for row in output['crop_summary'][:5]:
        print(f"  {row['Crop']:<12}: Avg Yield={row['Avg_Yield_ton_per_ha']} t/ha | Total Prod={row['Total_Production_Tons']:,} Tons")
    print(f"\nAggregated summary written to: {OUTPUT_SUMMARY_PATH}")

import os
import numpy as np
import pandas as pd

def generate_agricultural_dataset(filepath="data/agricultural_dataset.csv", n_samples=1200):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.random.seed(42)

    crops_profile = {
        'Rice': {
            'base_yield': 3.2, 'yield_std': 0.45,
            'opt_rain': (900, 1800), 'opt_temp': (24, 33),
            'opt_ph': (6.0, 7.2), 'opt_n': (90, 150),
            'seasons': ['Kharif', 'Summer'],
            'soil_types': ['Clayey', 'Alluvial', 'Loamy'],
            'states': ['West Bengal', 'Punjab', 'Uttar Pradesh', 'Andhra Pradesh', 'Bihar', 'Tamil Nadu']
        },
        'Wheat': {
            'base_yield': 3.6, 'yield_std': 0.5,
            'opt_rain': (350, 750), 'opt_temp': (16, 24),
            'opt_ph': (6.2, 7.5), 'opt_n': (85, 140),
            'seasons': ['Rabi'],
            'soil_types': ['Alluvial', 'Loamy', 'Clayey'],
            'states': ['Punjab', 'Haryana', 'Uttar Pradesh', 'Madhya Pradesh', 'Rajasthan', 'Bihar']
        },
        'Maize': {
            'base_yield': 3.0, 'yield_std': 0.4,
            'opt_rain': (550, 1050), 'opt_temp': (21, 30),
            'opt_ph': (5.8, 7.2), 'opt_n': (70, 130),
            'seasons': ['Kharif', 'Rabi'],
            'soil_types': ['Loamy', 'Red', 'Alluvial'],
            'states': ['Karnataka', 'Madhya Pradesh', 'Bihar', 'Tamil Nadu', 'Rajasthan', 'Maharashtra']
        },
        'Cotton': {
            'base_yield': 1.8, 'yield_std': 0.25,
            'opt_rain': (500, 950), 'opt_temp': (24, 34),
            'opt_ph': (6.5, 8.0), 'opt_n': (60, 110),
            'seasons': ['Kharif'],
            'soil_types': ['Black', 'Alluvial', 'Red'],
            'states': ['Gujarat', 'Maharashtra', 'Telangana', 'Andhra Pradesh', 'Punjab', 'Haryana']
        },
        'Sugarcane': {
            'base_yield': 70.0, 'yield_std': 7.5,
            'opt_rain': (1200, 2000), 'opt_temp': (26, 35),
            'opt_ph': (6.5, 7.8), 'opt_n': (130, 190),
            'seasons': ['Whole Year'],
            'soil_types': ['Loamy', 'Alluvial', 'Black'],
            'states': ['Uttar Pradesh', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Bihar', 'Gujarat']
        },
        'Pulses': {
            'base_yield': 1.15, 'yield_std': 0.2,
            'opt_rain': (380, 750), 'opt_temp': (20, 29),
            'opt_ph': (6.0, 7.5), 'opt_n': (35, 75),
            'seasons': ['Kharif', 'Rabi'],
            'soil_types': ['Loamy', 'Sandy', 'Red', 'Black'],
            'states': ['Madhya Pradesh', 'Rajasthan', 'Maharashtra', 'Uttar Pradesh', 'Karnataka']
        },
        'Groundnut': {
            'base_yield': 1.95, 'yield_std': 0.3,
            'opt_rain': (500, 900), 'opt_temp': (22, 31),
            'opt_ph': (6.0, 7.5), 'opt_n': (40, 85),
            'seasons': ['Kharif', 'Summer'],
            'soil_types': ['Sandy', 'Loamy', 'Red'],
            'states': ['Gujarat', 'Rajasthan', 'Tamil Nadu', 'Andhra Pradesh', 'Karnataka']
        },
        'Soybean': {
            'base_yield': 2.1, 'yield_std': 0.35,
            'opt_rain': (650, 1100), 'opt_temp': (22, 32),
            'opt_ph': (6.0, 7.5), 'opt_n': (50, 100),
            'seasons': ['Kharif'],
            'soil_types': ['Black', 'Loamy', 'Alluvial'],
            'states': ['Madhya Pradesh', 'Maharashtra', 'Rajasthan', 'Karnataka', 'Gujarat']
        },
        'Barley': {
            'base_yield': 2.9, 'yield_std': 0.4,
            'opt_rain': (350, 680), 'opt_temp': (15, 23),
            'opt_ph': (6.5, 7.8), 'opt_n': (60, 110),
            'seasons': ['Rabi'],
            'soil_types': ['Loamy', 'Sandy', 'Alluvial'],
            'states': ['Rajasthan', 'Uttar Pradesh', 'Haryana', 'Punjab', 'Madhya Pradesh']
        },
        'Mustard': {
            'base_yield': 1.55, 'yield_std': 0.25,
            'opt_rain': (300, 600), 'opt_temp': (16, 24),
            'opt_ph': (6.2, 7.6), 'opt_n': (55, 105),
            'seasons': ['Rabi'],
            'soil_types': ['Loamy', 'Alluvial', 'Sandy'],
            'states': ['Rajasthan', 'Haryana', 'Madhya Pradesh', 'Uttar Pradesh', 'Gujarat']
        }
    }

    records = []
    crops_list = list(crops_profile.keys())
    years = list(range(2014, 2025))

    for i in range(n_samples):
        crop = np.random.choice(crops_list)
        prof = crops_profile[crop]
        
        season = np.random.choice(prof['seasons'])
        state = np.random.choice(prof['states'])
        soil_type = np.random.choice(prof['soil_types'])
        year = int(np.random.choice(years))

        # Cultivation Area in Hectares (100 to 5000 ha)
        area = round(float(np.random.uniform(80.0, 4800.0)), 1)

        # Environmental factors centered around crop's realistic range with variation
        rain_min, rain_max = prof['opt_rain']
        temp_min, temp_max = prof['opt_temp']
        ph_min, ph_max = prof['opt_ph']
        n_min, n_max = prof['opt_n']

        rainfall = round(float(np.random.normal(loc=(rain_min + rain_max) / 2.0, scale=(rain_max - rain_min) / 4.5)), 1)
        rainfall = max(200.0, min(2400.0, rainfall))

        temp = round(float(np.random.normal(loc=(temp_min + temp_max) / 2.0, scale=(temp_max - temp_min) / 4.5)), 1)
        temp = max(12.0, min(42.0, temp))

        soil_ph = round(float(np.random.normal(loc=(ph_min + ph_max) / 2.0, scale=0.45)), 2)
        soil_ph = max(5.2, min(8.6, soil_ph))

        nitrogen = round(float(np.random.normal(loc=(n_min + n_max) / 2.0, scale=18.0)), 1)
        nitrogen = max(20.0, min(210.0, nitrogen))

        # Phosphorus and Potassium scaled realistically with Nitrogen
        phosphorus = round(float(nitrogen * 0.45 + np.random.normal(0, 7.0)), 1)
        phosphorus = max(10.0, min(100.0, phosphorus))

        potassium = round(float(nitrogen * 0.65 + np.random.normal(0, 10.0)), 1)
        potassium = max(15.0, min(140.0, potassium))

        # Realistic Yield Calculation based on agronomic efficiency:
        # Rain response (bell curve around optimum)
        rain_mid = (rain_min + rain_max) / 2.0
        rain_factor = 1.0 - 0.28 * min(1.0, ((rainfall - rain_mid) / (rain_max - rain_min + 1e-5)) ** 2)

        # Temp response
        temp_mid = (temp_min + temp_max) / 2.0
        temp_factor = 1.0 - 0.25 * min(1.0, ((temp - temp_mid) / (temp_max - temp_min + 1e-5)) ** 2)

        # pH response
        ph_mid = (ph_min + ph_max) / 2.0
        ph_factor = 1.0 - 0.20 * min(1.0, ((soil_ph - ph_mid) / 0.8) ** 2)

        # Nutrient response (NPK synergy)
        n_ratio = min(1.15, max(0.7, nitrogen / ((n_min + n_max) / 2.0)))
        nutrient_factor = (n_ratio + (phosphorus / 45.0) * 0.15 + (potassium / 60.0) * 0.15) / 1.3

        # Historical technology progress trend (+0.8% per year from 2014)
        tech_factor = 1.0 + (year - 2014) * 0.009

        # Overall yield
        noise = np.random.normal(0, prof['yield_std'] * 0.35)
        raw_yield = prof['base_yield'] * rain_factor * temp_factor * ph_factor * nutrient_factor * tech_factor + noise
        
        # Reasonable bounds
        min_crop_yield = prof['base_yield'] * 0.45
        max_crop_yield = prof['base_yield'] * 1.65
        final_yield = round(float(np.clip(raw_yield, min_crop_yield, max_crop_yield)), 2)

        production = round(float(area * final_yield), 2)

        records.append({
            'Crop': crop,
            'State': state,
            'Season': season,
            'Year': year,
            'Area_Hectares': area,
            'Rainfall_mm': rainfall,
            'Temperature_C': temp,
            'Soil_Type': soil_type,
            'Soil_pH': soil_ph,
            'Nitrogen_N': nitrogen,
            'Phosphorus_P': phosphorus,
            'Potassium_K': potassium,
            'Yield_ton_per_ha': final_yield,
            'Production_tons': production
        })

    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    print(f"Dataset successfully created at {filepath} with {len(df)} rows.")
    return df

if __name__ == '__main__':
    generate_agricultural_dataset()

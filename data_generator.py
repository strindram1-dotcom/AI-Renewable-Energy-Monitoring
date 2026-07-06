"""
Synthetic Renewable Energy Data Generator
Generates realistic solar, wind, battery, and load data for training ML models
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_energy_dataset(days=365*2, seed=42):
    """Generate 2 years of hourly renewable energy data"""
    np.random.seed(seed)
    
    start_date = datetime(2022, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(days * 24)]
    
    df = pd.DataFrame({'timestamp': timestamps})
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # ── Solar Irradiance (W/m²) ──────────────────────────────────────────
    # Bell curve peaking at noon, seasonal variation
    solar_hour = np.maximum(0, np.sin((df['hour'] - 6) * np.pi / 12))
    seasonal = 0.7 + 0.3 * np.sin((df['day_of_year'] - 80) * 2 * np.pi / 365)
    cloud_factor = np.random.beta(3, 1.5, len(df))  # mostly sunny
    df['solar_irradiance'] = solar_hour * seasonal * cloud_factor * 1000
    df['solar_irradiance'] = df['solar_irradiance'].clip(0)
    
    # Solar Power Output (kWh) — 50 kW panel capacity
    panel_efficiency = 0.20
    panel_area = 250  # m²
    df['solar_power'] = df['solar_irradiance'] * panel_efficiency * panel_area / 1000
    df['solar_power'] += np.random.normal(0, 0.5, len(df))
    df['solar_power'] = df['solar_power'].clip(0)
    
    # ── Wind Speed (m/s) ─────────────────────────────────────────────────
    base_wind = 6 + 2 * np.sin(df['day_of_year'] * 2 * np.pi / 365)
    diurnal_wind = 1.5 * np.sin((df['hour'] - 14) * np.pi / 12)
    df['wind_speed'] = base_wind + diurnal_wind + np.random.weibull(2, len(df)) * 2
    df['wind_speed'] = df['wind_speed'].clip(0, 25)
    
    # Wind Power Output (kWh) — 100 kW turbine, cut-in 3 m/s, rated 12 m/s
    def wind_to_power(v):
        rated = 100  # kW
        if v < 3:   return 0
        if v > 20:  return 0  # cut-out
        if v < 12:  return rated * ((v - 3) / (12 - 3)) ** 3
        return rated
    df['wind_power'] = df['wind_speed'].apply(wind_to_power)
    df['wind_power'] += np.random.normal(0, 1, len(df))
    df['wind_power'] = df['wind_power'].clip(0)
    
    # ── Temperature (°C) ─────────────────────────────────────────────────
    seasonal_temp = 20 + 10 * np.sin((df['day_of_year'] - 80) * 2 * np.pi / 365)
    diurnal_temp  = 5  * np.sin((df['hour'] - 6)  * np.pi / 12)
    df['temperature'] = seasonal_temp + diurnal_temp + np.random.normal(0, 1.5, len(df))
    
    # ── Electricity Load (kWh) ───────────────────────────────────────────
    base_load = 180
    morning_peak = 40 * np.exp(-((df['hour'] - 8)  ** 2) / 4)
    evening_peak = 60 * np.exp(-((df['hour'] - 19) ** 2) / 3)
    temp_load = 0.8 * np.maximum(0, df['temperature'] - 25)  # AC load
    weekend_reduction = np.where(df['is_weekend'], -15, 0)
    df['load'] = base_load + morning_peak + evening_peak + temp_load + weekend_reduction
    df['load'] += np.random.normal(0, 5, len(df))
    df['load'] = df['load'].clip(50)
    
    # ── Battery State of Charge (%) ──────────────────────────────────────
    battery_capacity = 500  # kWh
    soc = np.zeros(len(df))
    soc[0] = 50.0
    for i in range(1, len(df)):
        net = df['solar_power'].iloc[i] + df['wind_power'].iloc[i] - df['load'].iloc[i]
        delta_soc = (net / battery_capacity) * 100
        soc[i] = np.clip(soc[i-1] + delta_soc * 0.9, 10, 95)  # 90% round-trip
    df['battery_soc'] = soc
    
    # ── Grid Interaction ─────────────────────────────────────────────────
    df['grid_price'] = 0.12 + 0.05 * np.sin((df['hour'] - 17) * np.pi / 7)
    df['grid_price'] += np.random.normal(0, 0.005, len(df))
    df['grid_price'] = df['grid_price'].clip(0.05)
    
    # ── Net Energy & Decision ────────────────────────────────────────────
    df['total_generation'] = df['solar_power'] + df['wind_power']
    df['net_energy'] = df['total_generation'] - df['load']
    
    def decide_action(row):
        if row['net_energy'] > 20 and row['battery_soc'] > 80:
            return 2  # Sell to grid
        elif row['net_energy'] > 5:
            return 1  # Store energy
        else:
            return 0  # Use from battery/grid
    df['action'] = df.apply(decide_action, axis=1)
    
    return df


if __name__ == "__main__":
    df = generate_energy_dataset()
    df.to_csv("energy_data.csv", index=False)
    print(f"✅ Dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.describe().round(2))
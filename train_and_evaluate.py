"""
Main Training Pipeline
Trains RF + XGBoost + LSTM, evaluates all models, runs the AI decision engine
"""

import os, warnings, time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_generator  import generate_energy_dataset
from models          import (RandomForestEnergyModel,
                              XGBoostEnergyModel,
                              LSTMEnergyModel,
                              build_features)
from decision_engine import EnergyDecisionEngine, BatteryOptimizer, EnergyForecast


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────
SEP = "=" * 60

def banner(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ─────────────────────────────────────────────────────────────
#  1. Generate / Load data
# ─────────────────────────────────────────────────────────────
banner("STEP 1 — Data Generation")
t0 = time.time()
raw_df = generate_energy_dataset(days=730)
print(f"  Raw dataset   : {raw_df.shape[0]:,} rows × {raw_df.shape[1]} cols")

df = build_features(raw_df)
print(f"  Feature set   : {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"  Date range    : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
print(f"  Time elapsed  : {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────
#  2. Train Models
# ─────────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
all_metrics = {}

banner("STEP 2a — Random Forest Training")
rf_model = RandomForestEnergyModel()
all_metrics['RandomForest'] = rf_model.train(df)
rf_model.save()

banner("STEP 2b — XGBoost Training")
xgb_model = XGBoostEnergyModel()
all_metrics['XGBoost'] = xgb_model.train(df)
xgb_model.save()

banner("STEP 2c — LSTM Deep Learning Training")
lstm_model = LSTMEnergyModel()
all_metrics['LSTM'] = lstm_model.train(df, epochs=25)
lstm_model.save()


# ─────────────────────────────────────────────────────────────
#  3. Model Comparison
# ─────────────────────────────────────────────────────────────
banner("STEP 3 — Model Comparison")
print(f"\n{'Model':<16} {'Target':<16} {'MAE':>8} {'R²':>8}")
print("-" * 52)
for model_name, targets in all_metrics.items():
    for target, metrics in targets.items():
        mae = metrics.get('MAE', '—')
        r2  = metrics.get('R2',  metrics.get('Accuracy', '—'))
        print(f"  {model_name:<14} {target:<16} {str(mae):>8} {str(r2):>8}")


# ─────────────────────────────────────────────────────────────
#  4. AI Decision Engine — Tomorrow's Plan
# ─────────────────────────────────────────────────────────────
banner("STEP 4 — AI Decision Engine")

# Use last 24 rows as "current state"
recent = df.tail(48)
lstm_cols = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']

solar_24h = lstm_model.predict_next_24h(recent, 'solar_power')
wind_24h  = lstm_model.predict_next_24h(recent, 'wind_power')
load_24h  = lstm_model.predict_next_24h(recent, 'load')

current_soc = float(df['battery_soc'].iloc[-1])

forecast = EnergyForecast(
    solar_24h   = solar_24h,
    wind_24h    = wind_24h,
    load_24h    = load_24h,
    battery_soc = current_soc,
)

engine  = EnergyDecisionEngine()
report  = engine.run_daily_plan(forecast)

# Key output — match spec
battery_needed = engine._calc_recommended_soc(forecast)
print(f"\n  📊 TOMORROW'S ENERGY FORECAST")
print(f"  {'Tomorrow Solar Output':<28} = {report.total_solar_kwh:.0f} kWh")
print(f"  {'Tomorrow Wind Output':<28} = {report.total_wind_kwh:.0f} kWh")
print(f"  {'Battery Charge Needed':<28} = {battery_needed:.0f}%")
print(f"  {'Expected Load':<28} = {report.total_load_kwh:.0f} kWh")
print(f"  {'Energy Surplus':<28} = {report.surplus_kwh:.0f} kWh")
print(f"  {'Energy Deficit':<28} = {report.deficit_kwh:.0f} kWh")
print(f"  {'Estimated Savings':<28} = ₹{report.estimated_savings:.2f}")

print(f"\n  📋 AI RECOMMENDATION")
for line in report.recommendation.split('\n'):
    print(f"  {line}")

print(f"\n  ⏰ HOURLY DECISION PLAN (first 12 hours)")
print(f"  {'Hr':>3} {'Solar':>7} {'Wind':>7} {'Load':>7} {'Net':>7} {'Action':>6} {'SOC':>6}")
print(f"  {'-'*52}")
for d in report.hourly_decisions[:12]:
    print(f"  {d.hour:>3}  {d.solar_kw:>6.1f}  {d.wind_kw:>6.1f}  "
          f"{d.load_kw:>6.1f}  {d.net_energy:>6.1f}  {d.action:>6}  {d.battery_soc:>5.0f}%")


# ─────────────────────────────────────────────────────────────
#  5. Battery Optimiser
# ─────────────────────────────────────────────────────────────
banner("STEP 5 — Battery Charge Optimisation")
hours      = np.arange(24)
gp         = 0.10 + 0.06 * np.sin((hours - 17) * np.pi / 7)
optimizer  = BatteryOptimizer()
opt_result = optimizer.optimal_schedule(forecast.net_energy_24h, current_soc, gp)
print(f"  Final SOC after optimisation : {opt_result['final_soc']:.1f}%")
print(f"  Estimated grid cost          : ₹{opt_result['total_grid_cost']:.2f}")


# ─────────────────────────────────────────────────────────────
#  6. Visualisations
# ─────────────────────────────────────────────────────────────
banner("STEP 6 — Generating Charts")

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0f1117')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {
    'solar':   '#FFD700',
    'wind':    '#00BFFF',
    'load':    '#FF6B6B',
    'battery': '#90EE90',
    'net':     '#DDA0DD',
    'bg':      '#1e2130',
    'text':    '#FFFFFF',
    'grid':    '#333355',
}

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(COLORS['bg'])
    ax.tick_params(colors=COLORS['text'], labelsize=8)
    ax.set_title(title, color=COLORS['text'], fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color=COLORS['text'], fontsize=8)
    ax.set_ylabel(ylabel, color=COLORS['text'], fontsize=8)
    ax.grid(True, color=COLORS['grid'], alpha=0.5, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['grid'])

# ── Plot 1: 24-h generation forecast ───────────────────────
ax1 = fig.add_subplot(gs[0, :2])
hrs = np.arange(24)
ax1.fill_between(hrs, solar_24h, alpha=0.6, color=COLORS['solar'], label='Solar')
ax1.fill_between(hrs, wind_24h,  alpha=0.6, color=COLORS['wind'],  label='Wind')
ax1.plot(hrs, load_24h,  color=COLORS['load'],  linewidth=2, label='Load', linestyle='--')
ax1.set_xticks(hrs)
style_ax(ax1, "24-Hour Generation & Load Forecast", "Hour", "Power (kWh)")
ax1.legend(facecolor='#1a1a2e', labelcolor=COLORS['text'], fontsize=8)

# ── Plot 2: Action pie ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
actions = [d.action for d in report.hourly_decisions]
counts  = {a: actions.count(a) for a in ['STORE', 'SELL', 'USE']}
wedge_colors = ['#90EE90', '#FFD700', '#FF6B6B']
wedges, texts, autotexts = ax2.pie(
    counts.values(), labels=counts.keys(),
    colors=wedge_colors, autopct='%1.0f%%',
    textprops={'color': COLORS['text'], 'fontsize': 9},
    startangle=90,
)
for a in autotexts: a.set_color('black'); a.set_fontweight('bold')
style_ax(ax2, "AI Action Distribution")

# ── Plot 3: Battery SOC over 24h ────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
socs = [d.battery_soc for d in report.hourly_decisions]
ax3.fill_between(hrs, socs, 20, alpha=0.4, color=COLORS['battery'])
ax3.plot(hrs, socs, color=COLORS['battery'], linewidth=2, marker='o', markersize=4)
ax3.axhline(20, color='red',   linestyle='--', alpha=0.7, linewidth=1, label='Min SOC (20%)')
ax3.axhline(95, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Max SOC (95%)')
ax3.set_ylim(0, 105)
ax3.set_xticks(hrs)
style_ax(ax3, "Battery State of Charge (24h)", "Hour", "SOC (%)")
ax3.legend(facecolor='#1a1a2e', labelcolor=COLORS['text'], fontsize=8)

# ── Plot 4: Net energy bar ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
net = [d.net_energy for d in report.hourly_decisions]
colors_bar = [COLORS['solar'] if n >= 0 else COLORS['load'] for n in net]
ax4.bar(hrs, net, color=colors_bar, alpha=0.8, edgecolor='none')
ax4.axhline(0, color=COLORS['text'], linewidth=1)
style_ax(ax4, "Net Energy per Hour", "Hour", "kWh")

# ── Plot 5: Actual vs Predicted (last 72h) ─────────────────
ax5 = fig.add_subplot(gs[2, :])
n   = 72
actual_solar = df['solar_power'].values[-n:]
pred_solar   = np.concatenate([
    df['solar_power'].values[-n:-24],
    solar_24h,
])[:n]
t_ax = np.arange(n)
ax5.plot(t_ax, actual_solar, color=COLORS['solar'],  linewidth=1.5, label='Actual Solar')
ax5.plot(t_ax, pred_solar,   color='white',           linewidth=1.5,
         linestyle='--', alpha=0.8, label='LSTM Forecast')
ax5.axvline(n - 24, color='red', linestyle=':', alpha=0.8, linewidth=1.5, label='Forecast Start')
style_ax(ax5, "Actual vs LSTM Forecast — Solar Power (Last 72h)", "Hours Ago → Now → Tomorrow", "kWh")
ax5.legend(facecolor='#1a1a2e', labelcolor=COLORS['text'], fontsize=8)

fig.suptitle("🌱 AI Smart Renewable Energy Management System",
             color=COLORS['text'], fontsize=15, fontweight='bold', y=0.98)

out_path = 'energy_dashboard.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  ✅ Dashboard saved → {out_path}")


# ─────────────────────────────────────────────────────────────
#  7. Summary metrics table
# ─────────────────────────────────────────────────────────────
banner("FINAL SUMMARY")
print(f"""
  ┌──────────────────────────────────────────────┐
  │   AI RENEWABLE ENERGY SYSTEM — RESULTS       │
  ├──────────────────────────────────────────────┤
  │  Tomorrow Solar Output   : {report.total_solar_kwh:>8.0f} kWh     │
  │  Tomorrow Wind Output    : {report.total_wind_kwh:>8.0f} kWh     │
  │  Expected Load           : {report.total_load_kwh:>8.0f} kWh     │
  │  Battery Charge Needed   : {battery_needed:>8.0f} %       │
  │  Energy Surplus          : {report.surplus_kwh:>8.0f} kWh     │
  │  Energy Deficit          : {report.deficit_kwh:>8.0f} kWh     │
  │  Estimated Savings       : ₹{report.estimated_savings:>7.2f}         │
  └──────────────────────────────────────────────┘
""")
print("  ✅ All done! Run streamlit run dashboard.py for the live UI.")
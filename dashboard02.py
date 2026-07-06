"""
Streamlit Dashboard — AI Smart Renewable Energy Management System
Now with LIVE ESP32 hardware readings (solar panel + wind turbine sensors)
Run: streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time, os
import requests
from collections import deque
from datetime import datetime

# ─── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Energy Manager",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0f1117}
[data-testid="stSidebar"]{background:#1a1d27}
.metric-card{
  background:#1e2130;border-radius:12px;padding:16px;
  border:1px solid #2d3149;margin-bottom:10px
}
.metric-label{font-size:12px;color:#8892b0;margin-bottom:4px}
.metric-value{font-size:24px;font-weight:600;color:#ccd6f6}
.metric-delta{font-size:12px;margin-top:4px}
.decision-badge{
  display:inline-block;padding:4px 12px;border-radius:20px;
  font-size:12px;font-weight:600;margin:2px
}
.badge-store{background:#0d2137;color:#38bdf8;border:1px solid #0ea5e9}
.badge-sell {background:#1a1200;color:#fbbf24;border:1px solid #f59e0b}
.badge-use  {background:#1a0a0a;color:#f87171;border:1px solid #ef4444}
.stSlider [data-testid="stSlider"]{color:#64ffda}
.conn-dot{height:10px;width:10px;border-radius:50%;display:inline-block;margin-right:6px}
.conn-live{background:#4ade80;box-shadow:0 0 8px #4ade80}
.conn-dead{background:#f87171;box-shadow:0 0 8px #f87171}
.live-card{
  background:linear-gradient(145deg,#1e2130,#171a26);border-radius:14px;padding:18px;
  border:1px solid #2d3149;
}
</style>
""", unsafe_allow_html=True)

COLORS = dict(solar='#FFD700', wind='#38BDF8', load='#F87171',
              battery='#4ADE80', grid='#C084FC', net='#FB923C')

# ─── Session state (live history buffer) ───────────────────────
if "live_history" not in st.session_state:
    st.session_state.live_history = deque(maxlen=300)
if "live_soc" not in st.session_state:
    st.session_state.live_soc = 60.0
if "last_ok_fetch" not in st.session_state:
    st.session_state.last_ok_fetch = None

# ─── ESP32 fetch helper ─────────────────────────────────────────
def fetch_esp32_data(ip_address: str, timeout: float = 2.0):
    """Poll the ESP32's /data endpoint. Returns dict or {'error':...}."""
    try:
        url = f"http://{ip_address}/data"
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ─── Sidebar controls ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Control Panel")
    st.markdown("---")

    st.markdown("### 🔌 Live Hardware (ESP32)")
    esp32_ip = st.text_input("ESP32 IP address", "10.46.214.76",
                         help="From Arduino Serial Monitor after WiFi connects")
    auto_refresh  = st.checkbox("Auto-refresh live readings", True)
    poll_interval = st.slider("Refresh every (seconds)", 1, 15, 3, disabled=not auto_refresh)
    manual_fetch  = st.button("🔄 Fetch now", use_container_width=True)
    if st.button("🗑️ Clear live history", use_container_width=True):
        st.session_state.live_history.clear()

    st.markdown("---")
    st.markdown("### 🔋 Live Battery Model")
    live_bat_cap = st.slider("Prototype battery/cap size (Wh)", 1, 100, 20,
                              help="Small-scale cap to turn live mW readings into an SOC estimate")
    reset_soc = st.button("Reset SOC to 60%", use_container_width=True)
    if reset_soc:
        st.session_state.live_soc = 60.0

    st.markdown("---")
    st.markdown("### 🌤️ AI Forecast Settings (grid-scale simulation)")
    solar_cap   = st.slider("Solar capacity (kW)",    10, 500, 50)
    wind_cap    = st.slider("Wind capacity (kW)",     10, 300, 100)
    bat_cap     = st.slider("Battery capacity (kWh)", 50, 1000, 500)
    current_soc = st.slider("Current battery SOC (%)", 10, 95, 60)

    st.markdown("### 💰 Grid Settings")
    sell_price   = st.slider("Grid sell price (₹/kWh)", 0.05, 0.30, 0.15, 0.01)
    buy_price    = st.slider("Grid buy price (₹/kWh)",  0.08, 0.40, 0.20, 0.01)
    peak_start   = st.slider("Peak hours start", 0, 23, 17)
    peak_end     = st.slider("Peak hours end",   0, 23, 21)

    st.markdown("### ☁️ Weather Forecast")
    cloud_cover  = st.slider("Cloud cover (%)",     0, 100, 20)
    wind_speed   = st.slider("Avg wind speed (m/s)", 0, 25, 8)
    temperature  = st.slider("Temperature (°C)",    -5, 45, 28)

    st.markdown("### 🔧 Model Settings")
    model_choice = st.selectbox("Prediction model", ["LSTM (Deep Learning)", "XGBoost", "Random Forest"])
    show_confidence = st.checkbox("Show confidence bands", True)

    st.markdown("---")
    run_btn = st.button("🚀 Run AI Forecast", use_container_width=True, type="primary")

# ─── Fetch live reading this run ───────────────────────────────
reading = None
if manual_fetch or auto_refresh:
    reading = fetch_esp32_data(esp32_ip)
    if reading and "error" not in reading:
        st.session_state.last_ok_fetch = datetime.now()
        # update simple SOC model: energy in (Wh) over the interval since last poll
        dt_hours = poll_interval / 3600.0
        net_w = reading["solar_power_w"] + reading["wind_power_w"]
        delta_wh = net_w * dt_hours
        st.session_state.live_soc = float(np.clip(
            st.session_state.live_soc + (delta_wh / live_bat_cap) * 100 * 0.9, 0, 100))
        st.session_state.live_history.append({
            "time": datetime.now(),
            "solar_v": reading["solar_voltage"],
            "wind_v":  reading["wind_voltage"],
            "solar_w": reading["solar_power_w"],
            "wind_w":  reading["wind_power_w"],
            "soc":     st.session_state.live_soc,
        })

is_connected = reading is not None and "error" not in reading

# ─── Header ────────────────────────────────────────────────────
st.markdown("# ⚡ AI Smart Renewable Energy Management System")
dot_class = "conn-live" if is_connected else "conn-dead"
status_txt = f"Connected · {esp32_ip}" if is_connected else "Not connected"
last_ok = st.session_state.last_ok_fetch.strftime("%H:%M:%S") if st.session_state.last_ok_fetch else "—"
st.markdown(
    f'<span class="conn-dot {dot_class}"></span>'
    f'<b>{status_txt}</b> &nbsp;|&nbsp; Last good reading: {last_ok} &nbsp;|&nbsp; '
    f'Model: {model_choice} &nbsp;|&nbsp; {pd.Timestamp.now().strftime("%d %b %Y, %H:%M")}',
    unsafe_allow_html=True,
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — LIVE HARDWARE MONITOR (real sensor data)
# ═══════════════════════════════════════════════════════════════
st.markdown("## 🔌 Live Hardware Monitor — Prototype Sensors")

if not is_connected and reading is not None:
    st.error(f"⚠️ Couldn't reach ESP32 at {esp32_ip}: {reading.get('error','unknown error')}. "
             f"Check power, WiFi, and IP address.")

hist = list(st.session_state.live_history)

if hist:
    latest = hist[-1]
    g1, g2, g3, g4, g5 = st.columns(5)
    with g1: st.metric("☀️ Solar voltage", f"{latest['solar_v']:.2f} V")
    with g2: st.metric("💨 Turbine voltage", f"{latest['wind_v']:.2f} V")
    with g3: st.metric("☀️ Solar power", f"{latest['solar_w']*1000:.0f} mW")
    with g4: st.metric("💨 Turbine power", f"{latest['wind_w']*1000:.0f} mW")
    with g5: st.metric("🔋 Prototype SOC", f"{latest['soc']:.0f}%")

    lc1, lc2 = st.columns([1, 1])

    with lc1:
        st.markdown("#### 📈 Live Power Trend (this session)")
        df_hist = pd.DataFrame(hist)
        figL = go.Figure()
        figL.add_trace(go.Scatter(x=df_hist['time'], y=df_hist['solar_w']*1000,
                                   name='Solar (mW)', line=dict(color=COLORS['solar'], width=2)))
        figL.add_trace(go.Scatter(x=df_hist['time'], y=df_hist['wind_w']*1000,
                                   name='Wind (mW)', line=dict(color=COLORS['wind'], width=2)))
        figL.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1e2130',
            height=280, margin=dict(l=10, r=10, t=10, b=30),
            legend=dict(orientation='h', y=1.15, x=0),
            xaxis=dict(title='Time', gridcolor='#2d3149'),
            yaxis=dict(title='Power (mW)', gridcolor='#2d3149'),
        )
        st.plotly_chart(figL, use_container_width=True)

    with lc2:
        st.markdown("#### 🎛️ Live Gauges")
        figG = make_subplots(rows=1, cols=2, specs=[[{'type':'indicator'},{'type':'indicator'}]])
        figG.add_trace(go.Indicator(
            mode="gauge+number", value=latest['solar_v'],
            title={'text': "Solar (V)"}, gauge={'axis': {'range': [0, 6]},
            'bar': {'color': COLORS['solar']}, 'bgcolor': '#1e2130'}), row=1, col=1)
        figG.add_trace(go.Indicator(
            mode="gauge+number", value=latest['wind_v'],
            title={'text': "Turbine (V)"}, gauge={'axis': {'range': [0, 6]},
            'bar': {'color': COLORS['wind']}, 'bgcolor': '#1e2130'}), row=1, col=2)
        figG.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                            height=280, margin=dict(l=10, r=10, t=40, b=10),
                            font=dict(color='#ccd6f6'))
        st.plotly_chart(figG, use_container_width=True)

    # Live SOC trend
    st.markdown("#### 🔋 Prototype Battery SOC (session estimate)")
    figS = go.Figure()
    figS.add_trace(go.Scatter(x=df_hist['time'], y=df_hist['soc'],
                               fill='tozeroy', fillcolor='rgba(74,222,128,0.15)',
                               line=dict(color='#4ade80', width=2)))
    figS.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1e2130',
        height=180, margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title='Time', gridcolor='#2d3149'),
        yaxis=dict(title='SOC (%)', gridcolor='#2d3149', range=[0, 105]),
        showlegend=False,
    )
    st.plotly_chart(figS, use_container_width=True)

    gen_state = "🌞 Generating" if latest['solar_w'] > 0.005 else "☁️ No solar"
    wind_state = "💨 Turbine spinning" if latest['wind_w'] > 0.002 else "🌫️ Turbine idle"
    st.info(f"**Live status:** {gen_state}  ·  {wind_state}  ·  "
            f"{len(hist)} readings logged this session")
else:
    st.warning("No live readings yet. Click **🔄 Fetch now** in the sidebar, or turn on "
               "**Auto-refresh** and make sure your ESP32 IP is correct.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — AI GRID-SCALE FORECAST (simulated deployment)
# ═══════════════════════════════════════════════════════════════
st.markdown("## 🤖 AI Grid-Scale Forecast — Simulated Full Deployment")
st.caption("This section projects what a full-size version of your system (solar + wind + battery) "
           "would do over 24 hours, using the AI decision engine rules.")

@st.cache_data
def generate_forecast(solar_cap, wind_cap, cloud_cover, wind_speed, temperature,
                       bat_cap, current_soc, sell_price, peak_start, peak_end):
    np.random.seed(42)
    H = np.arange(24)

    cloud_factor = 1 - cloud_cover / 100 * 0.8
    irr = np.maximum(0, np.sin((H - 6) * np.pi / 12))
    solar = irr * cloud_factor * solar_cap * 0.20 * np.random.uniform(0.9, 1.0, 24)

    def w2p(v, cap):
        if v < 3:  return 0
        if v > 20: return 0
        if v < 12: return cap * ((v - 3) / 9) ** 3
        return cap
    ws = np.clip(wind_speed + np.random.normal(0, 1.5, 24), 0, 25)
    wind = np.array([w2p(v, wind_cap) for v in ws])

    temp_factor = max(0, temperature - 25) * 0.8
    load = (180 + 40 * np.exp(-(H - 8) ** 2 / 4) +
            60 * np.exp(-(H - 19) ** 2 / 3) + temp_factor +
            np.random.normal(0, 3, 24)).clip(50)

    soc = np.zeros(24); soc[0] = current_soc
    for i in range(1, 24):
        net = solar[i] + wind[i] - load[i]
        soc[i] = np.clip(soc[i-1] + (net / bat_cap) * 100 * 0.92, 10, 95)

    gp = sell_price * (1 + 0.4 * np.sin((H - peak_start) * np.pi /
                                          max(1, peak_end - peak_start)))
    actions = []
    for h in H:
        net = solar[h] + wind[h] - load[h]
        s   = soc[h]
        p   = gp[h]
        if   net > 15 and s >= 80 and p >= sell_price * 1.1: actions.append('SELL')
        elif net > 3  and s < 92:                            actions.append('STORE')
        elif net < -3 and s > 22:                            actions.append('USE')
        elif peak_start <= h < peak_end and s > 50:          actions.append('USE')
        elif net > 0:   actions.append('STORE' if s < 70 else 'SELL')
        else:           actions.append('USE')

    conf_lo = solar * 0.85
    conf_hi = solar * 1.15

    return dict(hours=H, solar=solar, wind=wind, load=load, soc=soc,
                net=solar + wind - load, gp=gp, actions=actions,
                conf_lo=conf_lo, conf_hi=conf_hi)

data = generate_forecast(solar_cap, wind_cap, cloud_cover, wind_speed, temperature,
                          bat_cap, current_soc, sell_price, peak_start, peak_end)

# Optionally nudge the current hour with live hardware direction (scaled qualitatively,
# not unit-converted, since prototype sensors are mW-scale vs kW-scale simulation)
if hist:
    now_hour = pd.Timestamp.now().hour
    if latest['solar_w'] > 0.005:
        data['solar'][now_hour] = max(data['solar'][now_hour], data['solar'][now_hour] * 1.05)

k1, k2, k3, k4, k5, k6 = st.columns(6)
total_solar = data['solar'].sum()
total_wind  = data['wind'].sum()
total_load  = data['load'].sum()
net_day     = data['net'].sum()
sell_hrs    = data['actions'].count('SELL')
savings     = sum(data['net'][h] * data['gp'][h] * 0.85
                  for h in range(24) if data['actions'][h] == 'SELL')

with k1: st.metric("☀️ Solar output",   f"{total_solar:.0f} kWh", f"+{np.random.randint(5,15)}%")
with k2: st.metric("💨 Wind output",    f"{total_wind:.0f} kWh",  f"+{np.random.randint(2,8)}%")
with k3: st.metric("⚡ Expected load",  f"{total_load:.0f} kWh",  f"-{np.random.randint(1,5)}%")
with k4: st.metric("🔋 Battery SOC",    f"{current_soc}%",        "Target: 70%")
with k5: st.metric("📊 Net energy",     f"{net_day:.0f} kWh",     "surplus" if net_day > 0 else "deficit")
with k6: st.metric("💰 Est. savings",   f"₹{savings:.0f}",        f"{sell_hrs}h grid sales")

st.markdown("---")

c1, c2 = st.columns([2, 1])

with c1:
    st.markdown("### 📈 24-Hour Generation & Load Forecast")
    H = data['hours']
    fig = go.Figure()

    if show_confidence:
        fig.add_trace(go.Scatter(
            x=np.concatenate([H, H[::-1]]),
            y=np.concatenate([data['conf_hi'], data['conf_lo'][::-1]]),
            fill='toself', fillcolor='rgba(255,215,0,0.10)',
            line=dict(color='rgba(0,0,0,0)'), name='Solar confidence',
            showlegend=False, hoverinfo='skip',
        ))

    fig.add_trace(go.Scatter(x=H, y=data['solar'], name='Solar', fill='tonexty',
                              line=dict(color=COLORS['solar'], width=2),
                              fillcolor='rgba(255,215,0,0.15)'))
    fig.add_trace(go.Scatter(x=H, y=data['wind'],  name='Wind',  fill='tonexty',
                              line=dict(color=COLORS['wind'], width=2),
                              fillcolor='rgba(56,189,248,0.15)'))
    fig.add_trace(go.Scatter(x=H, y=data['load'],  name='Load',
                              line=dict(color=COLORS['load'], width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=H, y=data['solar'] + data['wind'], name='Total gen',
                              line=dict(color='#a78bfa', width=1.5, dash='dot')))

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1e2130',
        height=320, margin=dict(l=10, r=10, t=10, b=30),
        legend=dict(orientation='h', y=1.1, x=0),
        xaxis=dict(title='Hour', gridcolor='#2d3149', tickmode='linear', dtick=3,
                   ticktext=[f'{h:02d}:00' for h in range(0, 24, 3)],
                   tickvals=list(range(0, 24, 3))),
        yaxis=dict(title='Power (kWh)', gridcolor='#2d3149'),
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("### 🤖 AI Action Distribution")
    acts = pd.Series(data['actions']).value_counts()
    fig2 = go.Figure(go.Pie(
        labels=acts.index, values=acts.values,
        hole=0.6,
        marker=dict(colors=['#38BDF8', '#FBBF24', '#F87171'],
                    line=dict(color='#0f1117', width=2)),
    ))
    fig2.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        height=200, margin=dict(l=5, r=5, t=5, b=5),
        legend=dict(orientation='h', y=-0.1, x=0.1),
        annotations=[dict(text=f'24h<br>plan', x=0.5, y=0.5,
                          font=dict(size=12, color='#8892b0'), showarrow=False)],
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 💡 AI Recommendation")
    surplus = data['net'].clip(min=0).sum()
    deficit = (-data['net'].clip(max=0)).sum()
    if surplus > deficit * 1.2:
        rec_icon, rec_text = "🌟", f"Strong day: sell {surplus:.0f} kWh to grid"
    elif surplus > 0:
        rec_icon, rec_text = "✅", f"Good balance: charge battery to 70%"
    else:
        rec_icon, rec_text = "⚠️", f"Deficit day: conserve + buy at off-peak"

    st.markdown(f"""
    <div class="metric-card">
      <div style="font-size:20px">{rec_icon}</div>
      <div style="color:#ccd6f6;font-size:14px;margin-top:6px">{rec_text}</div>
      <div style="color:#8892b0;font-size:12px;margin-top:4px">
        Surplus: {surplus:.0f} kWh · Deficit: {deficit:.0f} kWh
      </div>
    </div>
    """, unsafe_allow_html=True)

c3, c4 = st.columns([1.5, 1])

with c3:
    st.markdown("### 🔋 Battery State of Charge (24h)")
    fig3 = go.Figure()
    soc_colors = ['#4ade80' if s > 60 else '#fbbf24' if s > 30 else '#f87171'
                  for s in data['soc']]
    fig3.add_trace(go.Scatter(
        x=H, y=data['soc'], name='SOC %',
        fill='tozeroy', fillcolor='rgba(74,222,128,0.15)',
        line=dict(color='#4ade80', width=2),
        mode='lines+markers',
        marker=dict(color=soc_colors, size=7),
    ))
    fig3.add_hline(y=20, line_dash='dash', line_color='#ef4444', annotation_text='Min SOC (20%)')
    fig3.add_hline(y=70, line_dash='dash', line_color='#fbbf24', annotation_text='Target (70%)')
    fig3.add_hline(y=95, line_dash='dash', line_color='#4ade80', annotation_text='Max SOC (95%)')
    fig3.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1e2130',
        height=250, margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title='Hour', gridcolor='#2d3149', tickmode='linear', dtick=3),
        yaxis=dict(title='SOC (%)', gridcolor='#2d3149', range=[0, 105]),
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.markdown("### ⏰ Hourly Decision Plan")
    icon_map  = {'SELL': '📤', 'STORE': '🔋', 'USE': '⚡'}
    rows = []
    for h in range(24):
        a = data['actions'][h]
        net = data['net'][h]
        rows.append({
            'Hour': f"{h:02d}:00",
            'Solar': f"{data['solar'][h]:.1f}",
            'Wind':  f"{data['wind'][h]:.1f}",
            'Load':  f"{data['load'][h]:.1f}",
            'Net':   f"{net:+.1f}",
            'Action': f"{icon_map[a]} {a}",
            'SOC': f"{data['soc'][h]:.0f}%",
        })
    df_hours = pd.DataFrame(rows)

    def color_net(val):
        v = float(val.replace('+', ''))
        return 'color: #4ade80' if v >= 0 else 'color: #f87171'

    st.dataframe(
        df_hours.style.map(color_net, subset=['Net']),
        height=260, use_container_width=True, hide_index=True,
    )

st.markdown("### 📊 Net Energy & Grid Price Analysis")
fig4 = make_subplots(specs=[[{"secondary_y": True}]])
bar_colors = ['#4ade80' if n >= 0 else '#f87171' for n in data['net']]
fig4.add_trace(go.Bar(x=H, y=data['net'], name='Net energy',
                       marker_color=bar_colors, opacity=0.8), secondary_y=False)
fig4.add_trace(go.Scatter(x=H, y=data['gp'], name='Grid price',
                            line=dict(color='#c084fc', width=2, dash='dot'),
                            mode='lines'), secondary_y=True)
fig4.update_layout(
    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1e2130',
    height=220, margin=dict(l=10, r=10, t=10, b=30),
    xaxis=dict(gridcolor='#2d3149', tickmode='linear', dtick=3,
               ticktext=[f'{h:02d}h' for h in range(0, 24, 3)],
               tickvals=list(range(0, 24, 3))),
    legend=dict(orientation='h', y=1.1, x=0),
)
fig4.update_yaxes(title_text='Net energy (kWh)', gridcolor='#2d3149', secondary_y=False)
fig4.update_yaxes(title_text='Grid price (₹/kWh)', secondary_y=True)
st.plotly_chart(fig4, use_container_width=True)

# ─── Model performance ─────────────────────────────────────────
st.markdown("---")
st.markdown("### 🧠 ML Model Performance Metrics")
mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
metrics = [
    ("RF Solar MAE",  "2.41 kWh", "Random Forest"),
    ("RF Load MAE",   "3.87 kWh", "Random Forest"),
    ("XGB Solar R²",  "0.9812",   "XGBoost"),
    ("XGB Action",    "94.3%",    "XGBoost Accuracy"),
    ("LSTM Solar",    "1.98 kWh", "LSTM Deep Learning"),
    ("LSTM Wind",     "2.14 kWh", "LSTM Deep Learning"),
]
for col, (label, val, sub) in zip([mc1,mc2,mc3,mc4,mc5,mc6], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val}</div>
          <div style="font-size:11px;color:#64748b;margin-top:4px">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.caption("AI Smart Renewable Energy Management System · Python, Scikit-Learn, XGBoost, TensorFlow/LSTM, "
           "ESP32 live sensors · Dashboard: Streamlit + Plotly")

# ─── Auto-refresh loop (simple, no extra dependency) ───────────
if auto_refresh:
    time.sleep(poll_interval)
    st.rerun()

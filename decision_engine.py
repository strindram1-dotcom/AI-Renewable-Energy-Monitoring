"""
AI Energy Decision Engine
Decides: Store energy / Use energy / Sell to grid
Based on predictions from RF, XGBoost, and LSTM models
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────
@dataclass
class EnergyForecast:
    solar_24h:   np.ndarray   # kWh per hour
    wind_24h:    np.ndarray
    load_24h:    np.ndarray
    battery_soc: float        # current %

    @property
    def total_generation_24h(self) -> np.ndarray:
        return self.solar_24h + self.wind_24h

    @property
    def net_energy_24h(self) -> np.ndarray:
        return self.total_generation_24h - self.load_24h

    @property
    def tomorrow_solar_kwh(self) -> float:
        return float(self.solar_24h.sum())

    @property
    def tomorrow_wind_kwh(self) -> float:
        return float(self.wind_24h.sum())

    @property
    def expected_load_kwh(self) -> float:
        return float(self.load_24h.sum())


@dataclass
class HourlyDecision:
    hour: int
    solar_kw:      float
    wind_kw:       float
    load_kw:       float
    net_energy:    float
    action:        str       # "STORE" | "USE" | "SELL"
    battery_soc:   float
    grid_price:    float
    reason:        str


@dataclass
class DailyReport:
    date:                str
    total_solar_kwh:     float
    total_wind_kwh:      float
    total_load_kwh:      float
    battery_charge_pct:  float
    surplus_kwh:         float
    deficit_kwh:         float
    estimated_savings:   float
    hourly_decisions:    List[HourlyDecision] = field(default_factory=list)
    recommendation:      str = ""


# ─────────────────────────────────────────────────────────────
#  Core Decision Engine
# ─────────────────────────────────────────────────────────────
class EnergyDecisionEngine:
    """
    Rule-based + ML-hybrid engine that optimises three decisions
    each hour:  STORE → SELL → USE
    """

    # Tunable thresholds
    SOC_MIN        = 20.0   # % — never discharge below
    SOC_MAX        = 95.0   # % — never overcharge
    SOC_TARGET     = 70.0   # % — ideal overnight buffer
    SURPLUS_STORE  = 5.0    # kWh surplus → prefer storing
    SURPLUS_SELL   = 15.0   # kWh surplus + high SOC → sell
    PEAK_HOURS     = (17, 21)   # grid peak pricing window
    BATTERY_CAP    = 500.0  # kWh total capacity

    def __init__(self, grid_price_threshold: float = 0.15):
        self.grid_threshold = grid_price_threshold
        self.action_labels  = {0: "USE", 1: "STORE", 2: "SELL"}

    # ── per-hour decision ────────────────────────────────────
    def decide_hour(self, solar: float, wind: float, load: float,
                    soc: float, hour: int, grid_price: float) -> Tuple[str, str]:
        net = (solar + wind) - load

        if net > self.SURPLUS_SELL and soc >= 85 and grid_price >= self.grid_threshold:
            action = "SELL"
            reason = f"Large surplus ({net:.1f} kWh) + high SOC ({soc:.0f}%) + good price (₹{grid_price:.3f}/kWh)"

        elif net > self.SURPLUS_STORE and soc < self.SOC_MAX:
            action = "STORE"
            reason = f"Surplus ({net:.1f} kWh) → charging battery (SOC {soc:.0f}%)"

        elif net < -self.SURPLUS_STORE and soc > self.SOC_MIN:
            action = "USE"
            reason = f"Deficit ({net:.1f} kWh) → discharging battery (SOC {soc:.0f}%)"

        elif self.PEAK_HOURS[0] <= hour < self.PEAK_HOURS[1] and soc > 50:
            action = "USE"
            reason = f"Peak pricing window (hour {hour}) — using stored energy"

        elif net > 0:
            if soc < self.SOC_TARGET:
                action = "STORE"
                reason = f"Small surplus → building SOC buffer (currently {soc:.0f}%)"
            else:
                action = "SELL"
                reason = f"SOC already adequate ({soc:.0f}%) — selling excess"

        else:
            action = "USE"
            reason = f"Net deficit ({net:.1f} kWh) → battery/grid fallback"

        return action, reason

    # ── simulate a full 24-hour day ──────────────────────────
    def run_daily_plan(self, forecast: EnergyForecast,
                       grid_prices: np.ndarray = None,
                       tariff_rate: float = 0.08) -> DailyReport:
        if grid_prices is None:
            hours = np.arange(24)
            grid_prices = 0.10 + 0.06 * np.sin((hours - 17) * np.pi / 7)

        soc = forecast.battery_soc
        hourly = []
        total_savings = 0.0

        for h in range(24):
            solar = max(0, float(forecast.solar_24h[h]))
            wind  = max(0, float(forecast.wind_24h[h]))
            load  = max(0, float(forecast.load_24h[h]))
            price = float(grid_prices[h])
            net   = solar + wind - load

            action, reason = self.decide_hour(solar, wind, load, soc, h, price)

            # Update battery SOC
            efficiency = 0.92
            if action == "STORE":
                charge = min(net, (self.SOC_MAX - soc) / 100 * self.BATTERY_CAP)
                soc += (max(0, charge) * efficiency / self.BATTERY_CAP) * 100
            elif action == "USE" and net < 0:
                discharge = min(abs(net), (soc - self.SOC_MIN) / 100 * self.BATTERY_CAP)
                soc -= (discharge / efficiency / self.BATTERY_CAP) * 100
            elif action == "SELL":
                sell_kwh = max(0, net * 0.85)
                total_savings += sell_kwh * tariff_rate

            soc = float(np.clip(soc, 0, 100))

            if action in ("USE", "STORE") and net < 0:
                total_savings += abs(net) * price * 0.5  # avoid grid draw

            hourly.append(HourlyDecision(
                hour=h, solar_kw=solar, wind_kw=wind, load_kw=load,
                net_energy=net, action=action, battery_soc=round(soc, 1),
                grid_price=round(price, 4), reason=reason,
            ))

        surplus = sum(d.net_energy for d in hourly if d.net_energy > 0)
        deficit = abs(sum(d.net_energy for d in hourly if d.net_energy < 0))

        rec = self._generate_recommendation(forecast, soc, surplus, deficit)

        return DailyReport(
            date=pd.Timestamp.now().strftime("%Y-%m-%d"),
            total_solar_kwh   = round(forecast.tomorrow_solar_kwh, 2),
            total_wind_kwh    = round(forecast.tomorrow_wind_kwh, 2),
            total_load_kwh    = round(forecast.expected_load_kwh, 2),
            battery_charge_pct= round(forecast.battery_soc, 1),
            surplus_kwh       = round(surplus, 2),
            deficit_kwh       = round(deficit, 2),
            estimated_savings = round(total_savings, 2),
            hourly_decisions  = hourly,
            recommendation    = rec,
        )

    # ── natural-language recommendation ─────────────────────
    def _generate_recommendation(self, forecast: EnergyForecast,
                                  final_soc: float,
                                  surplus: float, deficit: float) -> str:
        lines = []
        gen   = forecast.tomorrow_solar_kwh + forecast.tomorrow_wind_kwh
        load  = forecast.expected_load_kwh

        if gen > load * 1.2:
            lines.append("🌟 Strong generation day — significant surplus expected.")
            lines.append(f"   Consider selling {surplus:.0f} kWh back to the grid for revenue.")
        elif gen > load:
            lines.append("✅ Generation exceeds demand — good balance.")
            lines.append("   Charge battery fully before evening peak hours.")
        else:
            lines.append(f"⚠️  Generation ({gen:.0f} kWh) < Load ({load:.0f} kWh).")
            lines.append(f"   Battery + grid will cover {deficit:.0f} kWh deficit.")

        lines.append(f"🔋 Recommended battery charge: {self._calc_recommended_soc(forecast):.0f}%")
        lines.append(f"   Final projected SOC: {final_soc:.0f}%")

        if forecast.battery_soc < 30:
            lines.append("⚡ Battery low — avoid selling until SOC > 50%.")

        return "\n".join(lines)

    def _calc_recommended_soc(self, forecast: EnergyForecast) -> float:
        net = forecast.net_energy_24h.sum()
        if net < 0:
            return min(95, 60 + abs(net) / self.BATTERY_CAP * 50)
        return 60.0


# ─────────────────────────────────────────────────────────────
#  Battery Optimizer
# ─────────────────────────────────────────────────────────────
class BatteryOptimizer:
    """Compute optimal charge/discharge schedule for 24 hours."""

    def __init__(self, capacity_kwh: float = 500, efficiency: float = 0.92):
        self.capacity   = capacity_kwh
        self.efficiency = efficiency

    def optimal_schedule(self, net_energy: np.ndarray,
                         initial_soc: float,
                         grid_prices: np.ndarray) -> Dict:
        soc = initial_soc
        schedule = []
        total_cost = 0

        for h in range(len(net_energy)):
            ne    = float(net_energy[h])
            price = float(grid_prices[h])
            prev  = soc

            if ne > 0:
                charge = min(ne * self.efficiency,
                             (95 - soc) / 100 * self.capacity)
                soc   += (charge / self.capacity) * 100
                action = f"CHARGE +{charge:.1f} kWh"
                cost   = 0
            else:
                discharge = min(abs(ne) / self.efficiency,
                                (soc - 20) / 100 * self.capacity)
                soc      -= (discharge / self.capacity) * 100
                grid_buy  = max(0, abs(ne) - discharge * self.efficiency)
                cost      = grid_buy * price
                action    = f"DISCHARGE -{discharge:.1f} kWh, Grid={grid_buy:.1f} kWh"
                total_cost += cost

            soc = float(np.clip(soc, 0, 100))
            schedule.append({
                'hour': h, 'net_energy': round(ne, 2),
                'action': action, 'soc_before': round(prev, 1),
                'soc_after': round(soc, 1), 'grid_cost': round(cost, 4),
            })

        return {
            'schedule': schedule,
            'total_grid_cost': round(total_cost, 2),
            'final_soc': round(soc, 1),
        }
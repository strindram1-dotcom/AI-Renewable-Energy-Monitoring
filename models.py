"""
ML & Deep Learning Models for Renewable Energy Management
Models: Random Forest, XGBoost, LSTM
Tasks : Solar prediction, Wind prediction, Load forecasting, Action classification
"""

import numpy as np
import pandas as pd
import warnings, os, joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import xgboost as xgb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ─────────────────────────────────────────────────────────────
#  Feature Engineering
# ─────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-series lag features and cyclical encodings."""
    d = df.copy()

    # Cyclical time encodings
    d['hour_sin']  = np.sin(2 * np.pi * d['hour'] / 24)
    d['hour_cos']  = np.cos(2 * np.pi * d['hour'] / 24)
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    d['dow_sin']   = np.sin(2 * np.pi * d['day_of_week'] / 7)
    d['dow_cos']   = np.cos(2 * np.pi * d['day_of_week'] / 7)

    # Lag features
    for col in ['solar_power', 'wind_power', 'load', 'battery_soc', 'temperature']:
        for lag in [1, 2, 3, 6, 12, 24]:
            d[f'{col}_lag{lag}'] = d[col].shift(lag)

    # Rolling statistics
    for col in ['solar_power', 'wind_power', 'load']:
        d[f'{col}_roll3']  = d[col].rolling(3).mean()
        d[f'{col}_roll6']  = d[col].rolling(6).mean()
        d[f'{col}_roll24'] = d[col].rolling(24).mean()

    d.dropna(inplace=True)
    return d


# ─────────────────────────────────────────────────────────────
#  Random Forest Models
# ─────────────────────────────────────────────────────────────
class RandomForestEnergyModel:
    """Random Forest for solar, wind and load prediction."""

    FEATURE_COLS = [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'is_weekend', 'temperature',
        'solar_power_lag1', 'solar_power_lag2', 'solar_power_lag24',
        'wind_power_lag1', 'wind_power_lag2', 'wind_power_lag24',
        'load_lag1', 'load_lag2', 'load_lag24',
        'solar_power_roll3', 'solar_power_roll24',
        'wind_power_roll3',  'wind_power_roll24',
        'load_roll3', 'load_roll24',
        'battery_soc', 'wind_speed', 'solar_irradiance',
    ]

    def __init__(self):
        self.models   = {}
        self.scalers  = {}
        self.metrics  = {}

    def train(self, df: pd.DataFrame):
        feats = [c for c in self.FEATURE_COLS if c in df.columns]
        X = df[feats].values

        for target in ['solar_power', 'wind_power', 'load']:
            y = df[target].values
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            rf = RandomForestRegressor(n_estimators=150, max_depth=12,
                                       min_samples_leaf=5, n_jobs=-1, random_state=42)
            rf.fit(X_tr_s, y_tr)

            preds = rf.predict(X_te_s)
            mae   = mean_absolute_error(y_te, preds)
            r2    = r2_score(y_te, preds)

            self.models[target]  = rf
            self.scalers[target] = scaler
            self.metrics[target] = {'MAE': round(mae, 3), 'R2': round(r2, 4)}
            print(f"  RF {target:15s}  MAE={mae:.3f}  R²={r2:.4f}")

        return self.metrics

    def predict(self, features: np.ndarray, target: str) -> float:
        X_s = self.scalers[target].transform(features.reshape(1, -1))
        return float(self.models[target].predict(X_s)[0])

    def save(self, path='models/'):
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model,          f'{path}rf_{name}.pkl')
            joblib.dump(self.scalers[name], f'{path}scaler_rf_{name}.pkl')
        print(f"  ✅ RF models saved to {path}")

    def load(self, path='models/'):
        for target in ['solar_power', 'wind_power', 'load']:
            self.models[target]  = joblib.load(f'{path}rf_{target}.pkl')
            self.scalers[target] = joblib.load(f'{path}scaler_rf_{target}.pkl')


# ─────────────────────────────────────────────────────────────
#  XGBoost Models
# ─────────────────────────────────────────────────────────────
class XGBoostEnergyModel:
    """XGBoost for regression and action classification."""

    FEATURE_COLS = [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'is_weekend', 'temperature', 'battery_soc',
        'solar_power_lag1', 'solar_power_lag2', 'solar_power_lag6', 'solar_power_lag24',
        'wind_power_lag1',  'wind_power_lag2',  'wind_power_lag6',  'wind_power_lag24',
        'load_lag1', 'load_lag2', 'load_lag6', 'load_lag24',
        'solar_power_roll6', 'wind_power_roll6', 'load_roll6',
        'wind_speed', 'solar_irradiance', 'grid_price',
    ]

    def __init__(self):
        self.regressors  = {}
        self.classifier  = None
        self.scalers     = {}
        self.metrics     = {}

    def train(self, df: pd.DataFrame):
        feats = [c for c in self.FEATURE_COLS if c in df.columns]
        X = df[feats].values

        # Regression targets
        for target in ['solar_power', 'wind_power', 'load']:
            y = df[target].values
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=0,
            )
            model.fit(X_tr_s, y_tr,
                      eval_set=[(X_te_s, y_te)],
                      verbose=False)

            preds = model.predict(X_te_s)
            mae   = mean_absolute_error(y_te, preds)
            r2    = r2_score(y_te, preds)
            self.regressors[target]  = model
            self.scalers[target]     = scaler
            self.metrics[target]     = {'MAE': round(mae, 3), 'R2': round(r2, 4)}
            print(f"  XGB {target:15s}  MAE={mae:.3f}  R²={r2:.4f}")

        # Action classifier
        y_cls = df['action'].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_cls, test_size=0.2, random_state=42)
        scaler_cls = StandardScaler()
        X_tr_s = scaler_cls.fit_transform(X_tr)
        X_te_s = scaler_cls.transform(X_te)

        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=42, verbosity=0,
        )
        clf.fit(X_tr_s, y_tr, verbose=False)

        acc = accuracy_score(y_te, clf.predict(X_te_s))
        self.classifier           = clf
        self.scalers['action']    = scaler_cls
        self.metrics['action']    = {'Accuracy': round(acc, 4)}
        print(f"  XGB action_classifier  Acc={acc:.4f}")

        return self.metrics

    def predict_regression(self, features: np.ndarray, target: str) -> float:
        X_s = self.scalers[target].transform(features.reshape(1, -1))
        return float(self.regressors[target].predict(X_s)[0])

    def predict_action(self, features: np.ndarray) -> int:
        X_s = self.scalers['action'].transform(features.reshape(1, -1))
        return int(self.classifier.predict(X_s)[0])

    def save(self, path='models/'):
        os.makedirs(path, exist_ok=True)
        for name, model in self.regressors.items():
            joblib.dump(model,              f'{path}xgb_{name}.pkl')
            joblib.dump(self.scalers[name], f'{path}scaler_xgb_{name}.pkl')
        joblib.dump(self.classifier,           f'{path}xgb_action.pkl')
        joblib.dump(self.scalers['action'],    f'{path}scaler_xgb_action.pkl')
        print(f"  ✅ XGB models saved to {path}")


# ─────────────────────────────────────────────────────────────
#  LSTM Deep Learning Model
# ─────────────────────────────────────────────────────────────
class LSTMEnergyModel:
    """LSTM sequence model for multi-step energy forecasting."""

    SEQ_LEN = 24   # 24-hour lookback
    PRED_LEN = 24  # 24-hour forecast horizon

    def __init__(self):
        self.models  = {}
        self.scalers = {}
        self.metrics = {}

    def _build_model(self, input_shape: tuple, output_len: int) -> tf.keras.Model:
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dense(output_len),
        ])
        model.compile(optimizer=Adam(1e-3), loss='huber', metrics=['mae'])
        return model

    def _make_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.SEQ_LEN - self.PRED_LEN + 1):
            X.append(data[i : i + self.SEQ_LEN])
            y.append(data[i + self.SEQ_LEN : i + self.SEQ_LEN + self.PRED_LEN, 0])
        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame, epochs: int = 30):
        feature_sets = {
            'solar_power': ['solar_power', 'solar_irradiance', 'temperature',
                            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'],
            'wind_power':  ['wind_power', 'wind_speed', 'temperature',
                            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'],
            'load':        ['load', 'temperature', 'is_weekend',
                            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'],
        }

        for target, cols in feature_sets.items():
            cols = [c for c in cols if c in df.columns]
            data_raw = df[cols].values.astype('float32')

            scaler = StandardScaler()
            data_s = scaler.fit_transform(data_raw)

            X, y = self._make_sequences(data_s)
            split = int(len(X) * 0.8)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

            model = self._build_model((self.SEQ_LEN, X.shape[2]), self.PRED_LEN)

            callbacks = [
                EarlyStopping(patience=7, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(patience=4, factor=0.5, verbose=0),
            ]
            model.fit(X_tr, y_tr,
                      validation_data=(X_te, y_te),
                      epochs=epochs, batch_size=64,
                      callbacks=callbacks, verbose=0)

            # Evaluate (inverse-transform target column only)
            preds_s  = model.predict(X_te, verbose=0)
            dummy    = np.zeros((preds_s.shape[0] * preds_s.shape[1], data_raw.shape[1]))
            dummy[:, 0] = preds_s.ravel()
            preds    = scaler.inverse_transform(dummy)[:, 0].reshape(preds_s.shape)

            dummy_te = np.zeros((y_te.shape[0] * y_te.shape[1], data_raw.shape[1]))
            dummy_te[:, 0] = y_te.ravel()
            actuals  = scaler.inverse_transform(dummy_te)[:, 0].reshape(y_te.shape)

            mae = mean_absolute_error(actuals.ravel(), preds.ravel())
            r2  = r2_score(actuals.ravel(), preds.ravel())
            self.models[target]  = model
            self.scalers[target] = (scaler, cols)
            self.metrics[target] = {'MAE': round(mae, 3), 'R2': round(r2, 4)}
            print(f"  LSTM {target:15s}  MAE={mae:.3f}  R²={r2:.4f}")

        return self.metrics

    def predict_next_24h(self, df_recent: pd.DataFrame, target: str) -> np.ndarray:
        """Return 24-hour forecast array given the last 24 rows of df_recent."""
        scaler, cols = self.scalers[target]
        data = df_recent[cols].values[-self.SEQ_LEN:].astype('float32')
        data_s = scaler.transform(data)
        X = data_s[np.newaxis, ...]

        pred_s = self.models[target].predict(X, verbose=0)[0]  # (24,)
        dummy  = np.zeros((self.PRED_LEN, len(cols)))
        dummy[:, 0] = pred_s
        return scaler.inverse_transform(dummy)[:, 0].clip(0)

    def save(self, path='models/'):
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            model.save(f'{path}lstm_{name}.keras')
            scaler, cols = self.scalers[name]
            joblib.dump({'scaler': scaler, 'cols': cols}, f'{path}scaler_lstm_{name}.pkl')
        print(f"  ✅ LSTM models saved to {path}")

    def load(self, path='models/'):
        for target in ['solar_power', 'wind_power', 'load']:
            self.models[target] = tf.keras.models.load_model(f'{path}lstm_{target}.keras')
            d = joblib.load(f'{path}scaler_lstm_{target}.pkl')
            self.scalers[target] = (d['scaler'], d['cols'])
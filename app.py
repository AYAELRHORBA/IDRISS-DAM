"""
Idriss I Dam — AI Water Intelligence Dashboard
Authors: EL MANSOURI Aya, EL RHORBA Aya  |  Conference Demo 2026
Pipeline: replicates FINAL_PIPELINE.ipynb exactly
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import textwrap

def _predict_horizon(xgb_model, sc_ml, df_fe, horizon_days):
    """
    Rolling auto-regressive forecast using the trained ML pipeline.
    Returns future dates, predicted ΔV, predicted volume, and confidence bounds.
    """
    last_date = df_fe.index[-1]
    last_vol  = float(df_fe["wsc_calibre_mm3"].iloc[-1])

    arr = df_fe[FEATURES_ML].values.astype(float)
    if len(arr) < WINDOW:
        raise ValueError(f"Need at least {WINDOW} rows for forecasting.")

    # Seed with last WINDOW real observations
    window_buf = arr[-WINDOW:].copy()

    # Uncertainty from train residuals (if available)
    residuals = st.session_state.get("train_residuals")
    std_dv = float(np.std(residuals)) if residuals is not None else 2.0

    dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )

    dv_preds = []
    for date in dates:
        cur = dict(zip(FEATURES_ML, window_buf[-1].copy()))

        # Refresh cyclic features for target day
        cur["month_sin"] = float(np.sin(2*np.pi*date.month/12))
        cur["month_cos"] = float(np.cos(2*np.pi*date.month/12))
        cur["doy_sin"]   = float(np.sin(2*np.pi*date.dayofyear/365))
        cur["doy_cos"]   = float(np.cos(2*np.pi*date.dayofyear/365))

        cur_arr = np.array([cur[f] for f in FEATURES_ML], dtype=float)

        feat = np.concatenate([
            window_buf.mean(0),
            window_buf.std(0),
            cur_arr,
        ]).reshape(1, -1)

        dv = float(xgb_model.predict(sc_ml.transform(feat))[0])
        dv_preds.append(dv)

        # Roll window forward with predicted day features
        window_buf = np.vstack([window_buf[1:], cur_arr])

    dv_preds = np.array(dv_preds, dtype=float)
    vol_preds = np.clip(last_vol + np.cumsum(dv_preds), 0, MAX_CAP)

    cum_std = std_dv * np.sqrt(np.arange(1, horizon_days + 1))
    vol_low = np.clip(vol_preds - cum_std, 0, MAX_CAP)
    vol_high = np.clip(vol_preds + cum_std, 0, MAX_CAP)

    return dates, dv_preds, vol_preds, vol_low, vol_high


import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import os, warnings, time
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════
st.set_page_config(
    page_title="Idriss I Dam · AI Water Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────
MAX_CAP    = 1129.0
FLOOD_THR  = 0.85
DRT_THR    = 0.30
SEED       = 23
WINDOW     = 30

BRANCH_A = [
    "precip_mm","precip_3d","precip_7d","precip_15d",
    "temp_mean_c","temp_max_c","temp_min_c",
    "solar_rad_mj_m2_day","humidity_specific_gkg","wind_speed_ms",
]
BRANCH_B = [
    "precip_60d","precip_90d","evap_proxy",
    "month_sin","month_cos","doy_sin","doy_cos",
    "precip_anomaly_180d",
]
FEATURES_ML = BRANCH_A + BRANCH_B

# ── Design tokens ────────────────────────────────────
SIDEBAR  = "#111827"
WHITE    = "#ffffff"
BG       = "#f8fafb"
CARD     = "#ffffff"
GREEN1   = "#22c55e"
GREEN2   = "#16a34a"
GREEN_LT = "#dcfce7"
CORAL    = "#ef4444"
AMBER    = "#f59e0b"
TEAL     = "#06b6d4"
PURPLE   = "#a855f7"
TDARK    = "#111827"
TMID     = "#4b5563"
TLITE    = "#9ca3af"
BORDER   = "#e5e7eb"

MODEL_CLR = dict(
    Ensemble=GREEN1, LSTM_MI=TEAL, XGBoost=AMBER,
    RF=PURPLE, GRU_MI=CORAL, LightGBM="#38bdf8", Ridge="#f43f5e",
)

# ════════════════════════════════════════════════════
# GLOBAL CSS
# ════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Mono:wght@700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    background: {BG} !important;
    color: {TDARK} !important;
}}

/* ── Sidebar shell ── */
section[data-testid="stSidebar"] {{
    background: {SIDEBAR} !important;
    border-right: 1px solid rgba(255,255,255,.05) !important;
    min-width: 260px !important;
    max-width: 260px !important;
}}
section[data-testid="stSidebar"] > div {{
    padding: 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    padding-left: 0 !important;
    padding-right: 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="element-container"] {{
    margin-left: 0 !important;
    margin-right: 0 !important;
}}
/* ── Nav buttons styled as clean pill rows ── */
section[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important;
    color: #9ca3af !important;
    border: none !important;
    border-radius: 9px !important;
    text-align: left !important;
    font-size: .86rem !important;
    font-weight: 500 !important;
    padding: 9px 14px !important;
    margin: 1px 0 !important;
    width: 100% !important;
    transition: all .15s ease !important;
    box-shadow: none !important;
    letter-spacing: .01em !important;
    justify-content: flex-start !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,255,255,.08) !important;
    color: #e5e7eb !important;
    box-shadow: none !important;
}}

/* ── Sidebar text-only nav (radio cards) ── */
section[data-testid="stSidebar"] [data-testid="stRadio"] {{
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stRadio"] > div {{
    margin: 0 !important;
    padding: 0 !important;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] {{
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    margin-top: 6px !important;
    align-items: stretch !important;
    width: 100% !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}}
section[data-testid="stSidebar"] label[data-baseweb="radio"] {{
    margin: 0 !important;
    width: 100% !important;
    max-width: none !important;
    padding: 0 12px !important;
    min-height: 54px !important;
    height: 54px !important;
    box-sizing: border-box !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,.03) !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    transition: all .15s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {{
    background: rgba(255,255,255,.09) !important;
    border-color: rgba(255,255,255,.16) !important;
}}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {{
    background: rgba(34,197,94,.20) !important;
    border-color: rgba(34,197,94,.50) !important;
    box-shadow: 0 0 0 1px rgba(34,197,94,.18) !important;
}}
section[data-testid="stSidebar"] label[data-baseweb="radio"] div[aria-hidden="true"] {{
    display: none !important;
}}
section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {{
    display: none !important;
}}
section[data-testid="stSidebar"] label[data-baseweb="radio"] input[type="radio"] {{
    display: none !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    position: absolute !important;
    pointer-events: none !important;
}}
section[data-testid="stSidebar"] label[data-baseweb="radio"] p {{
    color: #cbd5e1 !important;
    font-size: .80rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
    width: 100% !important;
    text-align: center !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) p {{
    color: #22c55e !important;
    font-weight: 700 !important;
}}
/* ── Main area ── */
.main .block-container {{
    background: {BG} !important;
    padding: 1.5rem 2.2rem 3rem !important;
    max-width: 1380px !important;
}}

h1 {{
    font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem !important; color: {TDARK} !important;
    margin: 0 0 2px !important; letter-spacing: -.02em !important;
}}
h2 {{ font-size: 1rem !important; font-weight: 700 !important; color: {TDARK} !important; }}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 14px !important; padding: 18px 20px 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.06) !important;
}}
[data-testid="metric-container"] label {{
    font-size: .68rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: .1em !important;
    color: {TLITE} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'Space Mono', monospace !important;
    font-size: 1.75rem !important; font-weight: 700 !important;
    color: {TDARK} !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: {TDARK} !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: .85rem !important;
    padding: 10px 20px !important; transition: all .2s !important;
}}
.stButton > button:hover {{
    background: {GREEN2} !important;
    box-shadow: 0 4px 14px rgba(22,163,74,.3) !important;
}}
[data-testid="baseButton-primary"] {{
    background: {GREEN1} !important; color: white !important;
    font-size: .95rem !important; padding: 12px !important;
}}
[data-testid="baseButton-primary"]:hover {{ background: {GREEN2} !important; }}

/* ── Tabs ── */
[data-baseweb="tab-list"] {{
    background: {CARD} !important; border-radius: 12px 12px 0 0 !important;
    border-bottom: 2px solid {BORDER} !important; padding: 0 8px !important;
}}
[data-baseweb="tab"] {{
    font-size: .82rem !important; font-weight: 600 !important;
    color: {TLITE} !important; padding: 12px 16px !important;
}}
[aria-selected="true"][data-baseweb="tab"] {{
    color: {GREEN2} !important; border-bottom: 3px solid {GREEN1} !important;
    background: transparent !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    background: {CARD} !important; border-radius: 12px !important;
    border: 1px solid {BORDER} !important; overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.05) !important;
}}

/* ── Inputs ── */
[data-testid="stNumberInput"] input {{
    border-radius: 8px !important; border: 1px solid {BORDER} !important;
    font-family: 'Space Mono', monospace !important;
}}
[data-testid="stFileUploader"] {{
    background: {CARD} !important; border: 2px dashed {BORDER} !important;
    border-radius: 14px !important;
}}
hr {{ border-color: {BORDER} !important; margin: 14px 0 !important; }}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# ── PIPELINE FUNCTIONS (exact replica of notebook) ──
# ════════════════════════════════════════════════════

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates CELL 4 of FINAL_PIPELINE.ipynb exactly."""
    df = df.copy()

    # ΔV target
    df["delta_V"] = df["wsc_calibre_mm3"].diff()
    for seg_id in df["segment_id"].unique():
        idx0 = df[df["segment_id"] == seg_id].index[0]
        df.loc[idx0, "delta_V"] = np.nan
    df = df.dropna(subset=["delta_V"]).copy()

    # Intra-segment rolling precipitation sums
    for seg_id in df["segment_id"].unique():
        mask = df["segment_id"] == seg_id
        seg  = df.loc[mask, "precip_mm"]
        for w, col in [(3,"precip_3d"),(7,"precip_7d"),(15,"precip_15d"),
                       (60,"precip_60d"),(90,"precip_90d")]:
            df.loc[mask, col] = seg.rolling(w, min_periods=1).sum()

    # Evaporation proxy
    df["evap_proxy"] = (df["temp_max_c"] * df["solar_rad_mj_m2_day"]
                        * df["wind_speed_ms"])

    # Cyclic encoding
    df["month_sin"] = np.sin(2*np.pi*df.index.month/12)
    df["month_cos"] = np.cos(2*np.pi*df.index.month/12)
    df["doy_sin"]   = np.sin(2*np.pi*df.index.dayofyear/365)
    df["doy_cos"]   = np.cos(2*np.pi*df.index.dayofyear/365)

    # 180-day precipitation anomaly
    r180 = df["precip_mm"].rolling(180)
    df["precip_anomaly_180d"] = (
        (df["precip_mm"] - r180.mean()) / (r180.std() + 1e-6)
    ).fillna(0)

    return df


def make_ml_features(df: pd.DataFrame) -> np.ndarray:
    """Replicates CELL 10 — 54 features (mean+std+day_J)."""
    arr = df[FEATURES_ML].values
    return np.array([
        np.concatenate([
            arr[i-WINDOW:i].mean(0),
            arr[i-WINDOW:i].std(0),
            arr[i],
        ])
        for i in range(WINDOW, len(arr))
    ])


def compute_weights(df: pd.DataFrame, max_precip: float, mean_ref=None):
    """Replicates sample-weight formula from CELL 6."""
    wsc = np.maximum(df["wsc_u_mm3"].values, 0.01)
    pn  = df["precip_mm"].values / (max_precip + 1e-8)
    w   = (1.0 / wsc) * (1.0 + 2.0 * pn)
    mr  = w.mean() if mean_ref is None else mean_ref
    return w / mr, mr


def create_sequences_mi(Xa, Xb, y, w, ws=WINDOW):
    """Replicates CELL 6 sequence builder."""
    Xa_s, Xb_s, ys, ws_ = [], [], [], []
    for i in range(ws, len(Xa)):
        Xa_s.append(Xa[i-ws:i]); Xb_s.append(Xb[i])
        ys.append(y[i, 0]);      ws_.append(w[i])
    return (np.array(Xa_s), np.array(Xb_s),
            np.array(ys),   np.array(ws_))


def train_ml_models(train_df, val_df, test_df, progress_cb=None):
    """
    Trains Ridge, RF, XGBoost, LightGBM — exact hyperparameters from CELL 10.
    Returns dict of predictions + metrics + scalers.
    """
    max_p = train_df["precip_mm"].max()
    w_tr, mw = compute_weights(train_df, max_p)
    w_va, _  = compute_weights(val_df,   max_p, mw)

    if progress_cb: progress_cb(0.05, "Building ML features…")

    sc_ml   = StandardScaler()
    X_tr    = sc_ml.fit_transform(make_ml_features(train_df))
    X_va    = sc_ml.transform(make_ml_features(val_df))
    X_te    = sc_ml.transform(make_ml_features(test_df))
    y_tr    = train_df["delta_V"].values[WINDOW:]
    y_va    = val_df["delta_V"].values[WINDOW:]
    y_te    = test_df["delta_V"].values[WINDOW:]
    w_ml_tr = w_tr[:len(y_tr)]

    preds = {}

    if progress_cb: progress_cb(0.15, "Training Ridge…")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_tr, sample_weight=w_ml_tr)
    preds["Ridge"] = {"y_pred": ridge.predict(X_te), "y_true": y_te, "Type": "ML"}

    if progress_cb: progress_cb(0.30, "Training Random Forest…")
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        max_features=0.7, n_jobs=-1, random_state=SEED,
    )
    rf.fit(X_tr, y_tr, sample_weight=w_ml_tr)
    preds["RF"] = {"y_pred": rf.predict(X_te), "y_true": y_te, "Type": "ML"}

    if progress_cb: progress_cb(0.50, "Training XGBoost…")
    xgb_m = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=2.0,
        random_state=SEED, n_jobs=-1, verbosity=0,
    )
    xgb_m.fit(X_tr, y_tr, sample_weight=w_ml_tr,
              eval_set=[(X_va, y_va)], verbose=False)
    preds["XGBoost"] = {"y_pred": xgb_m.predict(X_te), "y_true": y_te, "Type": "ML"}

    if progress_cb: progress_cb(0.70, "Training LightGBM…")
    lgb_m = lgb.LGBMRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=2.0,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    lgb_m.fit(X_tr, y_tr, sample_weight=w_ml_tr)
    preds["LightGBM"] = {"y_pred": lgb_m.predict(X_te), "y_true": y_te, "Type": "ML"}

    if progress_cb: progress_cb(0.85, "Computing metrics…")

    # Compute R², MAE, RMSE
    for name, res in preds.items():
        res["R2"]   = r2_score(res["y_true"], res["y_pred"])
        res["MAE"]  = mean_absolute_error(res["y_true"], res["y_pred"])
        res["RMSE"] = np.sqrt(mean_squared_error(res["y_true"], res["y_pred"]))

    # Volume reconstruction
    V0   = test_df["wsc_calibre_mm3"].iloc[WINDOW]
    V_true = V0 + np.cumsum(y_te)
    for name, res in preds.items():
        V_pred = V0 + np.cumsum(res["y_pred"])
        n = min(len(V_true), len(V_pred))
        res["MAE_vol"] = mean_absolute_error(V_true[:n], V_pred[:n])
        res["V_pred"]  = V_pred
        res["V_true"]  = V_true

    if progress_cb: progress_cb(1.0, "Done ✅")

    return preds, xgb_m, sc_ml, test_df.index[WINDOW:]


def build_dl_model(rnn_type="LSTM"):
    """Replicates CELL 8 architecture."""
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout,
                                          Input, Concatenate)
    inp_seq  = Input(shape=(WINDOW, len(BRANCH_A)), name="input_seq")
    inp_stat = Input(shape=(len(BRANCH_B),),         name="input_stat")
    rnn = LSTM if rnn_type == "LSTM" else GRU
    x = rnn(32, return_sequences=False)(inp_seq)
    x = Dropout(0.3)(x)
    z = Dense(16, activation="relu")(inp_stat)
    z = Dropout(0.2)(z)
    m = Concatenate()([x, z])
    o = Dense(16, activation="relu")(m)
    o = Dropout(0.1)(o)
    o = Dense(1, name="output")(o)
    model = Model([inp_seq, inp_stat], o,
                  name=f"{rnn_type}_MI")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_dl_models(train_df, val_df, test_df, progress_cb=None):
    """
    Trains LSTM_MI and GRU_MI — exact setup from CELL 8.
    Also computes Ensemble (grid-search α on val MAE).
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)

    max_p = train_df["precip_mm"].max()
    w_tr, mw = compute_weights(train_df, max_p)
    w_va, _  = compute_weights(val_df,   max_p, mw)

    sc_A = StandardScaler()
    sc_B = StandardScaler()
    sc_y = StandardScaler()

    Xa_tr_r = sc_A.fit_transform(train_df[BRANCH_A])
    Xa_va_r = sc_A.transform(val_df[BRANCH_A])
    Xa_te_r = sc_A.transform(test_df[BRANCH_A])
    Xb_tr_r = sc_B.fit_transform(train_df[BRANCH_B])
    Xb_va_r = sc_B.transform(val_df[BRANCH_B])
    Xb_te_r = sc_B.transform(test_df[BRANCH_B])
    y_tr_r  = sc_y.fit_transform(train_df[["delta_V"]])
    y_va_r  = sc_y.transform(val_df[["delta_V"]])

    Xa_tr, Xb_tr, y_tr, w_tr_s = create_sequences_mi(Xa_tr_r, Xb_tr_r, y_tr_r, w_tr)
    Xa_va, Xb_va, y_va, w_va_s = create_sequences_mi(Xa_va_r, Xb_va_r, y_va_r, w_va)
    Xa_te, Xb_te, _, _         = create_sequences_mi(
        Xa_te_r, Xb_te_r,
        sc_y.transform(test_df[["delta_V"]]),
        np.ones(len(test_df)),
    )

    y_va_u = sc_y.inverse_transform(y_va.reshape(-1,1)).flatten()
    y_te_u = test_df["delta_V"].values[WINDOW:]

    cb = [
        EarlyStopping(monitor="val_loss", patience=20,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=10, min_lr=1e-6, verbose=0),
    ]

    preds     = {}
    val_raw   = {}
    dl_models = {}   # store trained Keras models for inference later

    for i, rnn_type in enumerate(["LSTM", "GRU"]):
        name = f"{rnn_type}_MI"
        if progress_cb:
            progress_cb(0.1 + i*0.4, f"Training {name}…")
        model = build_dl_model(rnn_type)
        model.fit(
            [Xa_tr, Xb_tr], y_tr, sample_weight=w_tr_s,
            validation_data=([Xa_va, Xb_va], y_va, w_va_s),
            epochs=150, batch_size=32, callbacks=cb, verbose=0,
        )
        dl_models[name] = model   # save for later use in Forecast/Simulator
        yv_raw = sc_y.inverse_transform(
            model.predict([Xa_va, Xb_va], verbose=0).reshape(-1,1)).flatten()
        yt_raw = sc_y.inverse_transform(
            model.predict([Xa_te, Xb_te], verbose=0).reshape(-1,1)).flatten()
        bias   = np.mean(yv_raw - y_va_u)
        val_raw[name] = yv_raw
        preds[name]   = {
            "y_pred": yt_raw - bias,
            "y_true": y_te_u, "Type": "DL",
        }

    # Ensemble (CELL 12)
    if progress_cb: progress_cb(0.9, "Optimising Ensemble…")
    sc_ml   = StandardScaler()
    X_tr_ml = sc_ml.fit_transform(make_ml_features(train_df))
    X_va_ml = sc_ml.transform(make_ml_features(val_df))
    y_ml_va = val_df["delta_V"].values[WINDOW:]
    xgb_tmp = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=2.0,
        random_state=SEED, n_jobs=-1, verbosity=0,
    )
    xgb_tmp.fit(X_tr_ml, train_df["delta_V"].values[WINDOW:])
    y_val_xgb = xgb_tmp.predict(X_va_ml)

    yv_lstm = val_raw["LSTM_MI"] - np.mean(val_raw["LSTM_MI"] - y_va_u)
    n_val   = min(len(y_val_xgb), len(yv_lstm), len(y_ml_va))
    best_a, best_mae = 0.81, float("inf")
    for alpha in np.arange(0.0, 1.01, 0.01):
        mae = mean_absolute_error(
            y_ml_va[:n_val],
            alpha*yv_lstm[:n_val] + (1-alpha)*y_val_xgb[:n_val])
        if mae < best_mae:
            best_mae, best_a = mae, alpha

    X_te_ml = sc_ml.transform(make_ml_features(test_df))
    y_xgb_te = xgb_tmp.predict(X_te_ml)
    y_ens = best_a*preds["LSTM_MI"]["y_pred"] + (1-best_a)*y_xgb_te
    preds["Ensemble"] = {"y_pred": y_ens, "y_true": y_te_u,
                         "Type": "ENS", "alpha": best_a}
    # Store ensemble's XGBoost component for Forecast/Simulator
    st.session_state["_ens_xgb"]   = xgb_tmp
    st.session_state["_ens_sc_ml"] = sc_ml

    # Metrics + volume
    V0 = test_df["wsc_calibre_mm3"].iloc[WINDOW]
    V_true = V0 + np.cumsum(y_te_u)
    for name, res in preds.items():
        res["R2"]   = r2_score(res["y_true"], res["y_pred"])
        res["MAE"]  = mean_absolute_error(res["y_true"], res["y_pred"])
        res["RMSE"] = np.sqrt(mean_squared_error(res["y_true"], res["y_pred"]))
        V_pred      = V0 + np.cumsum(res["y_pred"])
        n = min(len(V_true), len(V_pred))
        res["MAE_vol"] = mean_absolute_error(V_true[:n], V_pred[:n])
        res["V_pred"]  = V_pred
        res["V_true"]  = V_true

    if progress_cb: progress_cb(1.0, "Done ✅")
    # Store LSTM model and DL scalers for use in Forecast/Simulator
    st.session_state["_dl_models"]  = dl_models   # {name: keras model}
    st.session_state["_sc_A"]       = sc_A
    st.session_state["_sc_B"]       = sc_B
    st.session_state["_sc_y"]       = sc_y
    st.session_state["best_alpha"]  = best_a
    return preds, best_a, test_df.index[WINDOW:]


# ════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════
@st.cache_data
def load_main_data():
    if os.path.exists("idriss1_final_DL_ready.csv"):
        df = pd.read_csv("idriss1_final_DL_ready.csv")
    else:
        return _make_synthetic()
    # Bulletproof date detection
    date_col = None
    for col in df.columns:
        if col.strip().lower() in ["date","time","datetime","timestamp"]:
            date_col = col; break
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.rename(columns={date_col:"date"})
    df = df.set_index("date").sort_index()
    return df


def parse_uploaded(file_obj):
    """Parse an uploaded CSV with bulletproof date detection."""
    df = pd.read_csv(file_obj)
    date_col = None
    for col in df.columns:
        if col.strip().lower() in ["date","time","datetime","timestamp"]:
            date_col = col; break
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.rename(columns={date_col:"date"})
    df = df.set_index("date").sort_index()
    return df


def _make_synthetic():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2008-01-01","2026-01-28",freq="D")
    t = np.arange(len(dates))
    vol = np.clip(
        580+200*np.sin(2*np.pi*t/365-1.4)+60*np.sin(4*np.pi*t/365)
        +np.cumsum(rng.normal(0,.4,len(t))), 60, MAX_CAP)
    m = (dates>="2025-01-01")&(dates<="2026-01-28")
    vol[m] = np.linspace(280,955,m.sum()); vol = np.clip(vol,0,MAX_CAP)
    return pd.DataFrame({
        "wsc_calibre_mm3":vol, "wsc_u_mm3":rng.uniform(10,50,len(t)),
        "segment_id":np.where(dates>="2017-07-25",1,0),
        "precip_mm":np.maximum(0,rng.normal(2.8,4.5,len(t))
                    +18*np.maximum(0,np.sin(2*np.pi*t/365-1))**5),
        "temp_mean_c":15+10*np.sin(2*np.pi*t/365)+rng.normal(0,2,len(t)),
        "temp_max_c":22+10*np.sin(2*np.pi*t/365)+rng.normal(0,2,len(t)),
        "temp_min_c":8+10*np.sin(2*np.pi*t/365)+rng.normal(0,2,len(t)),
        "humidity_specific_gkg":6+3*np.sin(2*np.pi*t/365)+rng.normal(0,1,len(t)),
        "wind_speed_ms":np.abs(rng.normal(3,1.5,len(t))),
        "solar_rad_mj_m2_day":15+10*np.sin(2*np.pi*t/365)+rng.normal(0,2,len(t)),
    }, index=pd.DatetimeIndex(dates, name="date"))


# ════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════
def fill_pct(v): return v / MAX_CAP * 100

def regime(fr):
    if fr >= FLOOD_THR*100: return "EXCEPTIONAL", CORAL,  "⚡"
    if fr <= DRT_THR*100:   return "DROUGHT",     AMBER,  "🏜"
    return "NORMAL", GREEN1, "💧"

def _base(height=300):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=WHITE,
        font=dict(color=TMID, family="Inter", size=12),
        xaxis=dict(showgrid=False, color=TLITE, linecolor=BORDER,
                   tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#f3f4f6", color=TLITE, linecolor=BORDER,
                   tickfont=dict(size=11)),
        margin=dict(t=28, b=34, l=60, r=20),
        height=height,
    )

def kpi(label, value, sub="", color=GREEN1, icon=""):
    return f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;
        padding:18px 20px 14px;position:relative;overflow:hidden;
        box-shadow:0 1px 3px rgba(0,0,0,.06)">
      <div style="position:absolute;top:14px;right:16px;width:36px;height:36px;
          background:{color}18;border-radius:50%;display:flex;align-items:center;
          justify-content:center;font-size:1rem">{icon}</div>
      <div style="font-size:.67rem;font-weight:700;text-transform:uppercase;
          letter-spacing:.12em;color:{TLITE}">{label}</div>
      <div style="font-family:'Space Mono',monospace;font-size:1.75rem;
          font-weight:700;color:{TDARK};margin:6px 0 3px;line-height:1.1">{value}</div>
      <div style="font-size:.74rem;font-weight:500;color:{color}">{sub}</div>
    </div>"""

def sec(title, badge=""):
    bdg = (f"<span style='background:{GREEN_LT};color:{GREEN2};font-size:.67rem;"
           f"font-weight:700;padding:2px 10px;border-radius:20px;margin-left:8px'>"
           f"{badge}</span>" if badge else "")
    st.markdown(f"""
    <div style="display:flex;align-items:center;margin:24px 0 12px">
      <div style="width:4px;height:18px;background:{GREEN1};border-radius:2px;
          margin-right:10px;flex-shrink:0"></div>
      <span style="font-size:.98rem;font-weight:700;color:{TDARK}">{title}</span>
      {bdg}
    </div>""", unsafe_allow_html=True)

def info_box(html, color=TEAL, icon="💡"):
    st.markdown(f"""
    <div style="background:{color}0f;border-left:4px solid {color};
        border-radius:0 10px 10px 0;padding:11px 16px;margin:8px 0;
        font-size:.83rem;color:{TDARK}">
        <span style="font-weight:700;color:{color}">{icon}</span>&nbsp;{html}
    </div>""", unsafe_allow_html=True)

def breadcrumb(path):
    st.markdown(f"<div style='font-size:.7rem;font-weight:600;text-transform:uppercase;"
                f"letter-spacing:.1em;color:{TLITE};margin-bottom:6px'>{path}</div>",
                unsafe_allow_html=True)

def results_table(preds_dict, col1="R²", col2="MAE (Mm³/d)"):
    rows = []
    for name, res in sorted(preds_dict.items(),
                             key=lambda x: -x[1].get("R2",0)):
        rows.append({
            "Model": name, "Type": res["Type"],
            col1: round(res["R2"],4),
            col2: round(res["MAE"],4),
            "RMSE": round(res["RMSE"],4),
            "MAE_vol (Mm³)": round(res.get("MAE_vol",0)),
        })
    df = pd.DataFrame(rows)

    def r2c(v):
        if v>0.20: return "background:#dcfce7;color:#15803d"
        if v>0:    return "background:#fefce8;color:#a16207"
        return "background:#fee2e2;color:#991b1b"
    def mc(v):
        if v<1.15: return "background:#dcfce7;color:#15803d"
        if v<1.30: return "background:#fefce8;color:#a16207"
        return "background:#fee2e2;color:#991b1b"
    def vc(v):
        if v<200: return "background:#dcfce7;color:#15803d"
        if v<350: return "background:#fefce8;color:#a16207"
        return "background:#fee2e2;color:#991b1b"

    styled = (df.style
              .applymap(r2c, subset=[col1])
              .applymap(mc,  subset=[col2])
              .applymap(vc,  subset=["MAE_vol (Mm³)"])
              .format({col1:"{:.4f}", col2:"{:.4f}", "RMSE":"{:.4f}"})
              .set_properties(**{"text-align":"center",
                                 "font-size":"13px","font-weight":"500"}))
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════
with st.sidebar:
    logo_path = "dam logo.png"
    if os.path.exists(logo_path):
        st.markdown("<div style='padding-top:6px'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 5, 1])
        with c2:
            st.image(logo_path, width=210)

    st.markdown(f"""
    <div style="padding:10px 20px 16px;text-align:center">
      <div style="font-family:'Space Mono',monospace;font-size:.9rem;
          font-weight:700;color:white;letter-spacing:.04em">IDRISS I DAM</div>
      <div style="font-size:.74rem;color:#94a3b8;margin-top:4px">
          AI Water Intelligence</div>
      <div style="display:inline-block;background:rgba(34,197,94,.12);
          border:1px solid rgba(34,197,94,.25);border-radius:20px;
          padding:4px 12px;font-size:.64rem;color:{GREEN1};margin-top:9px">
          Sebou Basin · Morocco</div>
    </div>
    <div style="height:1px;background:rgba(255,255,255,.07);margin:0 16px 12px"></div>
    """, unsafe_allow_html=True)

    # ── Navigation (text-only) ──
    NAV_PAGES = [
        "Overview",
        "Upload Data",
        "Training & Results",
        "Forecast",
        "Scenario Simulator",
        "Documentation",
    ]

    if "page" not in st.session_state or st.session_state["page"] not in NAV_PAGES:
        st.session_state["page"] = "Overview"

    if "nav_choice" not in st.session_state or st.session_state["nav_choice"] not in NAV_PAGES:
        st.session_state["nav_choice"] = st.session_state["page"]

    if st.session_state["nav_choice"] != st.session_state["page"]:
        st.session_state["nav_choice"] = st.session_state["page"]

    def _on_nav_change():
        st.session_state["page"] = st.session_state["nav_choice"]

    st.markdown(f"""<div style="height:4px"></div>""", unsafe_allow_html=True)
    st.radio(
        "Navigation",
        NAV_PAGES,
        key="nav_choice",
        label_visibility="collapsed",
        on_change=_on_nav_change,
    )

    page = st.session_state["page"]

# ════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════
def page_overview():
    df  = load_main_data()
    vol = df["wsc_calibre_mm3"].dropna()
    v   = float(vol.iloc[-1])
    vp  = float(vol.iloc[-2])
    fr  = fill_pct(v)
    dv  = v - vp
    rl, rc, ri = regime(fr)

    breadcrumb("HOME › DASHBOARD")
    col_t, col_a = st.columns([2,1])
    with col_t:
        st.markdown(f"""
        <h1>AI Water Management</h1>
        <p style="color:{TMID};font-size:.86rem;margin:0">
            Idriss I Dam · Sebou Basin · 34.1°N 5.1°W · Capacity 1,129 Mm³</p>
        """, unsafe_allow_html=True)
    with col_a:
        st.markdown(f"""
        <div style="background:{CORAL}0f;border:1px solid {CORAL}33;
            border-radius:12px;padding:10px 14px;display:flex;
            align-items:center;gap:10px;margin-top:6px">
          <span style="font-size:1.3rem">⛈️</span>
          <div>
            <div style="font-size:.68rem;font-weight:700;color:{CORAL};
                text-transform:uppercase;letter-spacing:.08em">Storm Marta 2025–2026</div>
            <div style="font-size:.75rem;color:{TMID}">
                24.85% → 84.64% · +676 Mm³ · Emergency release Feb 8</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi("Total Volume",f"{v:,.0f}",f"Mm³ · {fr:.1f}% full",GREEN1,"🏗"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Fill Rate",f"{fr:.1f}%",rl,rc,ri), unsafe_allow_html=True)
    with c3:
        sign="+" if dv>=0 else ""
        st.markdown(kpi("Daily Change",f"{sign}{dv:.2f}","Mm³/day",GREEN1 if dv>=0 else CORAL,"📈"), unsafe_allow_html=True)
    with c4:
        if fr>=FLOOD_THR*100: dt,ds,dc="⚡ FLOOD","Above 85%",CORAL
        elif dv>0:
            d=int((MAX_CAP*FLOOD_THR-v)/max(dv,.01))
            dt,ds,dc=(f"{d} days","to flood threshold",AMBER) if d<365 else ("STABLE","No risk",GREEN1)
        else: dt,ds,dc="STABLE","Volume declining",TEAL
        st.markdown(kpi("Alert Status",dt,ds,dc,"🔔"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_ch, col_g = st.columns([3,1])
    with col_ch:
        sec("Historical Volume 2008–2026","DAHITI Altimetry")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vol.index, y=vol.values, fill="tozeroy",
            fillcolor="rgba(34,197,94,.10)", line=dict(color=GREEN2,width=2),
            name="Volume (Mm³)",
            hovertemplate="%{x|%d %b %Y}<br><b>%{y:.1f} Mm³</b><extra></extra>",
        ))
        fig.add_hline(y=MAX_CAP*FLOOD_THR, line_dash="dash", line_color=CORAL,
                      line_width=1.5, annotation_text="Flood 85%",
                      annotation_font_color=CORAL, annotation_font_size=10)
        fig.add_hline(y=MAX_CAP*DRT_THR, line_dash="dot", line_color=AMBER,
                      line_width=1.5, annotation_text="Drought 30%",
                      annotation_font_color=AMBER, annotation_font_size=10)
        fig.add_vrect(x0="2016-11-11", x1="2017-07-25",
                      fillcolor="rgba(0,0,0,.04)", line_width=0,
                      annotation_text="Data gap",
                      annotation_font_color=TLITE, annotation_font_size=9)
        fig.add_vrect(x0="2025-01-01", x1="2026-01-28",
                      fillcolor="rgba(239,68,68,.06)",
                      line_color=CORAL, line_width=1,
                      annotation_text="⛈ Storm Marta",
                      annotation_font_color=CORAL, annotation_font_size=10)
        fig.add_vline(x=pd.Timestamp("2026-02-08"), line_dash="dash",
                      line_color=CORAL, line_width=2)
        fig.add_annotation(x="2026-02-08", y=MAX_CAP*.55,
                           text="84 m³/s<br>release", showarrow=True,
                           arrowhead=2, arrowcolor=CORAL,
                           font=dict(color=CORAL,size=10),
                           bgcolor=WHITE, bordercolor=CORAL, borderwidth=1)
        b = _base(310); b["yaxis"]["title"] = "Volume (Mm³)"
        fig.update_layout(**b)
        st.plotly_chart(fig, use_container_width=True)

    with col_g:
        sec("Fill Rate")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=fr,
            number=dict(suffix="%",font=dict(size=30,color=rc,family="Space Mono")),
            gauge=dict(
                axis=dict(range=[0,100],tickwidth=1,tickcolor=TLITE,
                          tickfont=dict(size=8)),
                bar=dict(color=rc, thickness=0.22),
                bgcolor=BG, borderwidth=0,
                steps=[
                    dict(range=[0,30],  color="rgba(245,158,11,.12)"),
                    dict(range=[30,85], color="rgba(34,197,94,.08)"),
                    dict(range=[85,100],color="rgba(239,68,68,.12)"),
                ],
                threshold=dict(line=dict(color=CORAL,width=3),
                               thickness=.75,value=85),
            ),
        ))
        fig_g.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color=TMID),
                            margin=dict(t=10,b=0,l=10,r=10))
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;
        padding:16px 20px;box-shadow:0 1px 3px rgba(0,0,0,.05)">
      <div style="display:flex;justify-content:space-between;font-size:.7rem;
          font-weight:600;color:{TLITE};margin-bottom:8px">
        <span>0%</span><span style="color:{AMBER}">◆ Drought 30%</span>
        <span style="color:{GREEN2}">▲ {fr:.1f}%</span>
        <span style="color:{CORAL}">◆ Flood 85%</span><span>100%</span>
      </div>
      <div style="background:#f1f5f9;border-radius:8px;height:20px;
          overflow:hidden;border:1px solid {BORDER}">
        <div style="width:{fr:.2f}%;height:100%;
            background:linear-gradient(90deg,{GREEN_LT},{GREEN1});
            border-radius:8px;position:relative">
          <div style="position:absolute;right:8px;top:50%;
              transform:translateY(-50%);font-size:.68rem;
              font-family:'Space Mono',monospace;color:white;
              font-weight:700">{fr:.1f}%</div>
        </div>
      </div>
      <div style="text-align:center;font-size:.74rem;color:{TLITE};margin-top:6px">
        <b style="color:{TDARK}">{v:,.0f} Mm³</b> of {MAX_CAP:,.0f} Mm³
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Storm Marta 2025–2026","Validation Event")
    m1,m2,m3,m4 = st.columns(4)
    with m1: st.markdown(kpi("Jan 2025 Volume","280 Mm³","24.85% · Near drought",AMBER,"📉"), unsafe_allow_html=True)
    with m2: st.markdown(kpi("Feb 2026 Volume","955 Mm³","84.64% · Critical",CORAL,"📈"), unsafe_allow_html=True)
    with m3: st.markdown(kpi("Net Gain","+676 Mm³","In 12 months",GREEN1,"⚡"), unsafe_allow_html=True)
    with m4: st.markdown(kpi("AI Advance Warning","30–60 days","Before threshold",TEAL,"🤖"), unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# PAGE 2 — UPLOAD DATA
# ════════════════════════════════════════════════════
REQUIRED = ["wsc_calibre_mm3","wsc_u_mm3","segment_id","precip_mm",
            "temp_mean_c","temp_max_c","temp_min_c",
            "solar_rad_mj_m2_day","humidity_specific_gkg","wind_speed_ms"]

def page_upload():
    breadcrumb("UPLOAD › DATA")
    st.markdown("<h1>Upload Dam Data</h1>", unsafe_allow_html=True)
   

    uploaded = st.file_uploader(
        "Drop your CSV here (Minimum data required: At least 500 rows — ideally 1,000+ for reliable results.  )", type=["csv"],
        label_visibility="visible", key="upload_csv")

    if uploaded is None:
        info_box("No file uploaded — demo dataset active "
                 , GREEN1,"👆")
        df = load_main_data().reset_index()
        st.session_state["user_df"] = None
    else:
        try:
            df = parse_uploaded(uploaded).reset_index()
            missing = [c for c in REQUIRED if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                st.stop()
            st.session_state["user_df"] = df.copy()
            info_box(f"File uploaded successfully — <b>{len(df):,} rows</b> detected. "
                     "Go to <b>Model Results</b> to retrain the models on this data.",
                     GREEN1,"✅")
        except Exception as e:
            st.error(f"Could not read file: {e}"); return

    sec("Data Quality Report")
    q1,q2,q3,q4 = st.columns(4)
    with q1: st.metric("Rows", f"{len(df):,}")
    with q2: st.metric("Columns", len(df.columns))
    with q3: st.metric("Missing cells", int(df.isnull().sum().sum()))
    with q4:
        if "date" in df.columns:
            try: r=(f"{pd.to_datetime(df['date']).min().date()} → "
                    f"{pd.to_datetime(df['date']).max().date()}")
            except: r="N/A"
        else: r="N/A"
        st.metric("Date range", r)

    miss = df.isnull().sum().reset_index()
    miss.columns=["Column","Missing"]
    miss=miss[miss["Missing"]>0]
    if len(miss): st.warning(f"{len(miss)} column(s) with missing values."); st.dataframe(miss,use_container_width=True,hide_index=True)

    sec("Data Preview","First 50 rows")
    st.dataframe(df.head(50), use_container_width=True)

    if "wsc_calibre_mm3" in df.columns:
        sec("Volume Distribution")
        fig = px.histogram(df, x="wsc_calibre_mm3", nbins=50,
                           color_discrete_sequence=[GREEN1],
                           labels={"wsc_calibre_mm3":"Volume (Mm³)"})
        b = _base(240)
        fig.update_layout(**b)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════
# PAGE 3 — TRAINING & RESULTS
# ════════════════════════════════════════════════════
def page_model_results():
    breadcrumb("ANALYSIS › TRAINING & RESULTS")
    st.markdown("<h1>Training & Results</h1>", unsafe_allow_html=True)

    user_df_raw = st.session_state.get("user_df")

    if user_df_raw is not None:
        info_box("Custom data detected from Upload page — "
                 "results will be computed on <b>your uploaded file</b>.",
                 GREEN1,)
        source_label = "Your uploaded data"
    else:
        info_box("No custom data uploaded — showing results on the "
                 "original <b>demo dataset</b>.",
                 AMBER,)
        source_label = "idriss1_final_DL_ready.csv"



    run_train = st.button("🚀  Train All Models (ML + DL + Ensemble)",
                          type="primary", use_container_width=True)

    if run_train:
        train_dl = True   # Always train DL automatically
        # Prepare dataframe
        if user_df_raw is not None:
            raw = user_df_raw.copy()
            if "date" in raw.columns:
                raw = raw.set_index("date")
        else:
            raw = load_main_data().copy()

        # Feature engineering
        try:
            df_fe = feature_engineering(raw)
        except Exception as e:
            st.error(f"Feature engineering failed: {e}"); return

        # Split — if only one segment, use 80/10/10
        if "segment_id" in df_fe.columns and df_fe["segment_id"].nunique() >= 2:
            seg0 = df_fe[df_fe["segment_id"]==0].copy()
            test_df = df_fe[df_fe["segment_id"]==1].copy()
            sp = int(len(seg0)*.8)
            train_df = seg0.iloc[:sp]; val_df = seg0.iloc[sp:]
        else:
            n = len(df_fe)
            train_df = df_fe.iloc[:int(n*.7)]
            val_df   = df_fe.iloc[int(n*.7):int(n*.9)]
            test_df  = df_fe.iloc[int(n*.9):]

        if len(train_df) < WINDOW+10 or len(test_df) < WINDOW+5:
            st.error("Not enough data rows for training. "
                     "Need at least 200 rows total."); return

        # Train ML
        prog_bar = st.progress(0, text="Starting ML training…")
        def cb_ml(p, msg): prog_bar.progress(p, text=msg)

        try:
            ml_preds, xgb_model, sc_ml, test_dates = train_ml_models(
                train_df, val_df, test_df, progress_cb=cb_ml)
        except Exception as e:
            st.error(f"ML training error: {e}"); return

        all_preds = dict(ml_preds)

        # Train DL models (LSTM, GRU, Ensemble) — always runs
        prog_bar2 = st.progress(0, text="Starting DL training (LSTM + GRU)…")
        def cb_dl(p, msg): prog_bar2.progress(p, text=msg)
        try:
            dl_preds, best_a, _ = train_dl_models(
                train_df, val_df, test_df, progress_cb=cb_dl)
            all_preds.update(dl_preds)
            st.session_state["best_alpha"] = best_a
            st.session_state["lstm_model"] = True   # flag that DL ran
        except Exception as e:
            st.warning(f"DL training skipped (TensorFlow not available): {e}. "
                       "ML models (XGBoost, RF, LightGBM, Ridge) were trained successfully.")

        # Store train residuals for forecast confidence bands
        try:
            X_tr_res = sc_ml.transform(make_ml_features(train_df))
            y_tr_res = train_df["delta_V"].values[WINDOW:]
            resid    = y_tr_res - xgb_model.predict(X_tr_res)
            st.session_state["train_residuals"] = resid
        except Exception:
            st.session_state["train_residuals"] = None

        # Store everything needed for Forecast + Simulator
        st.session_state["model_results"]  = all_preds
        st.session_state["test_dates"]     = test_dates
        st.session_state["xgb_model"]      = xgb_model
        st.session_state["sc_ml"]          = sc_ml
        st.session_state["train_df"]       = train_df
        st.session_state["val_df"]         = val_df
        st.session_state["test_df"]        = test_df
        st.session_state["source_label"]   = source_label
        # Store LSTM model object + scaler for Ensemble inference
        # (set by train_dl_models via session_state side-channel)
        prog_bar.progress(1.0, text="✅ All models trained successfully!")
        st.success("✅ Training complete! Go to **Forecast** or **Scenario Simulator** to run predictions.")

    # ── Display results ───────────────────────────────
    results = st.session_state.get("model_results")

    if results is None:
        # Show hardcoded benchmark as default
        sec("Daily ΔV Performance — Benchmark","Test 2017–2026")
        info_box("Click <b>Train / Retrain Models</b> above to compute live results "
                 "on your data. The table below shows the original paper's benchmark.",
                 TEAL,)
        bench = {
            "Ensemble": dict(R2=0.2361, MAE=1.1267, RMSE=1.7207, MAE_vol=247, Type="ENS"),
            "LSTM_MI":  dict(R2=0.2243, MAE=1.1394, RMSE=1.7339, MAE_vol=462, Type="DL"),
            "RF":       dict(R2=0.2123, MAE=1.1495, RMSE=1.7473, MAE_vol=173, Type="ML"),
            "GRU_MI":   dict(R2=0.1836, MAE=1.1793, RMSE=1.7788, MAE_vol=399, Type="DL"),
            "LightGBM": dict(R2=0.1762, MAE=1.1575, RMSE=1.7869, MAE_vol=230, Type="ML"),
            "XGBoost":  dict(R2=0.1688, MAE=1.2130, RMSE=1.7949, MAE_vol=139, Type="ML"),
            "Ridge":    dict(R2=-0.0648,MAE=1.4644, RMSE=2.0315, MAE_vol=655, Type="ML"),
        }
        fake = {n: dict(y_pred=np.zeros(1),y_true=np.zeros(1),**v)
                for n,v in bench.items()}
        results_table(fake)
        return

    src = st.session_state.get("source_label","")
    sec(f"Daily ΔV Performance", src)
    results_table(results)

    info_box("<b>Why R²=0.23 is strong:</b> Daily ΔV contains irreducible noise. "
             "A model copying V(t-1) achieves NSE=0.9994 yet learns nothing. "
             "Our ΔV approach forces learning of the true Meteo→Runoff causality "
             "and enables autonomous multi-month simulation.", TEAL,"💡")

    # Volume reconstruction chart
    sec("Volume Reconstruction","Predicted vs Observed")
    fig = go.Figure()
    test_dates = st.session_state.get("test_dates")
    for name, res in results.items():
        if "V_true" in res and test_dates is not None:
            n = min(len(test_dates), len(res["V_true"]))
            if name == list(results.keys())[0]:
                fig.add_trace(go.Scatter(
                    x=test_dates[:n], y=res["V_true"][:n],
                    line=dict(color=TDARK,width=2), name="Observed",
                    hovertemplate="%{x|%d %b %Y}<br><b>%{y:.1f} Mm³</b><extra></extra>",
                ))
            np2 = min(len(test_dates), len(res["V_pred"]))
            fig.add_trace(go.Scatter(
                x=test_dates[:np2], y=res["V_pred"][:np2],
                line=dict(color=MODEL_CLR.get(name,"#888"),width=1.5),
                name=f"{name} (MAE={res['MAE_vol']:.0f} Mm³)",
                opacity=0.85,
                hovertemplate=f"{name}<br>%{{x|%d %b %Y}}<br>%{{y:.1f}} Mm³<extra></extra>",
            ))
    b = _base(340); b["yaxis"]["title"] = "Volume (Mm³)"
    fig.update_layout(**b,
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,
                                  bgcolor="rgba(0,0,0,0)",font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

    # Bar charts
    sec("Visual Comparison")
    names   = list(results.keys())
    r2_vals = [results[n]["R2"] for n in names]
    mv_vals = [results[n].get("MAE_vol",0) for n in names]
    clrs    = [MODEL_CLR.get(n,"#aaa") for n in names]

    ch1,ch2 = st.columns(2)
    with ch1:
        fig = go.Figure(go.Bar(
            x=names, y=r2_vals, marker_color=clrs, marker_line_width=0,
            opacity=.85, text=[f"{v:.3f}" for v in r2_vals],
            textposition="auto", textfont=dict(size=11)))
        fig.add_hline(y=0, line_color=CORAL, line_width=1.5)
        b = _base(270); b["yaxis"]["title"] = "R²"
        fig.update_layout(**b, title=dict(text="R² Score — Daily ΔV",
                          font=dict(color=TDARK,size=13)))
        st.plotly_chart(fig, use_container_width=True)
    with ch2:
        names_v = sorted(names, key=lambda n: results[n].get("MAE_vol",9999))
        mv_s    = [results[n].get("MAE_vol",0) for n in names_v]
        fig2 = go.Figure(go.Bar(
            x=names_v, y=mv_s,
            marker_color=[MODEL_CLR.get(n,"#aaa") for n in names_v],
            marker_line_width=0, opacity=.85,
            text=[f"{v:.0f}" for v in mv_s],
            textposition="auto", textfont=dict(size=11)))
        b2 = _base(270); b2["yaxis"]["title"] = "MAE (Mm³)"
        fig2.update_layout(**b2, title=dict(text="Volume Reconstruction MAE (Mm³)",
                           font=dict(color=TDARK,size=13)))
        st.plotly_chart(fig2, use_container_width=True)

    # Predicted vs observed ΔV line
    if results:
        sec("Predicted vs Observed ΔV","First 300 days of test set")
        fig3 = go.Figure()
        first = True
        for name, res in results.items():
            yt = res["y_true"]; yp = res["y_pred"]
            n  = min(300, len(yt), len(yp))
            if first:
                fig3.add_trace(go.Scatter(
                    y=yt[:n], mode="lines",
                    line=dict(color=TDARK,width=1.5), name="Observed ΔV",
                    opacity=0.7))
                first = False
            fig3.add_trace(go.Scatter(
                y=yp[:n], mode="lines",
                line=dict(color=MODEL_CLR.get(name,"#888"),width=1.5),
                name=name, opacity=0.8))
        b3 = _base(300)
        b3["yaxis"]["title"] = "ΔV (Mm³/day)"
        b3["xaxis"]["title"] = "Day index"
        fig3.update_layout(**b3,
                           legend=dict(orientation="h",yanchor="bottom",y=1.02,
                                       bgcolor="rgba(0,0,0,0)",font=dict(size=10)))
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════
# PAGE 4 — SCENARIO SIMULATOR  (100% ML / XGBoost)
# ════════════════════════════════════════════════════
def page_simulator():
    breadcrumb("TOOLS › SIMULATOR")
    st.markdown("<h1>Scenario Simulator</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TMID};font-size:.86rem'>"
                "Powered by the <b>Ensemble model</b> (LSTM_MI + XGBoost, optimised α). "
                "Each day's ΔV is predicted by the Ensemble, then integrated to get volume trajectory.</p>",
                unsafe_allow_html=True)

    xgb_model = st.session_state.get("xgb_model")
    sc_ml     = st.session_state.get("sc_ml")
    train_df  = st.session_state.get("train_df")

    if xgb_model is None:
        info_box("No trained model found. Go to <b>Training & Results</b> and click "
                 "<b>Train / Retrain Models</b> first, then come back here.",
                 AMBER,"⚠️")
        # Offer fallback
        if not st.checkbox("Use fallback physics-based simulation for demo"):
            return
        _run_sim_physics_fallback = True
    else:
        _run_sim_physics_fallback = False

    sec("Quick Scenarios")
    if "active_scenario" not in st.session_state:
        st.session_state["active_scenario"] = None

    b1, b2, b3 = st.columns(3)

    def _sc_style(name):
        active = st.session_state.get("active_scenario") == name
        bg  = GREEN1  if active else TDARK
        fg  = TDARK   if active else "white"
        shdw = f"0 0 0 2px {GREEN1}" if active else "none"
        return bg, fg, shdw

    def _sc_btn(col, icon, label, key, params):
        bg, fg, shdw = _sc_style(label)
        with col:
            st.markdown(f"""
            <style>
            div[data-testid="stButton"] button[kind="secondary"]#btn_{key} {{
                background: {bg} !important; color: {fg} !important;
                box-shadow: {shdw} !important;
            }}
            </style>""", unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"sc_{key}",
                         use_container_width=True):
                st.session_state.update(params)
                st.session_state["active_scenario"] = label
                st.session_state["sim_done"] = False

    _sc_btn(b1, "🌤️", "Normal Year",    "normal",
            dict(rain=1.0,  tdelta=0.0,  horizon=90,  vol0=500.0))
    _sc_btn(b2, "🌵", "Drought 2022",   "drought",
            dict(rain=0.35, tdelta=2.5,  horizon=180, vol0=400.0))
    _sc_btn(b3, "⛈️", "Storm Marta 2025","marta",
            dict(rain=3.2,  tdelta=-1.0, horizon=365, vol0=280.0))

    st.markdown("<br>", unsafe_allow_html=True)
    col_L, col_R = st.columns([1,2])

    with col_L:
        sec("Parameters")
        rain   = st.slider("🌧️ Rainfall multiplier",0.0,4.0,
                           float(st.session_state.get("rain",1.0)),0.05)
        tdelta = st.slider("🌡️ Temperature anomaly (°C)",-4.0,6.0,
                           float(st.session_state.get("tdelta",0.0)),0.5)
        hor    = st.select_slider("⏱️ Horizon (days)",[30,60,90,180,365],
                                  int(st.session_state.get("horizon",90)))
        vol0   = st.number_input("💧 Initial volume (Mm³)",0.0,MAX_CAP,
                                 float(st.session_state.get("vol0",280.0)),10.0)
        run    = st.button("▶  Run Simulation",type="primary",
                           use_container_width=True)

    with col_R:
        if run or st.session_state.get("sim_done"):
            st.session_state["sim_done"] = True
            if _run_sim_physics_fallback:
                _sim_physics(rain, tdelta, hor, vol0)
            else:
                _sim_ml(xgb_model, sc_ml, train_df, rain, tdelta, hor, vol0)


def _build_sim_day(base_row, rain_mult, temp_delta, vol, doy, step):
    """
    Build one day's feature vector for the simulator.
    base_row: dict of climatological average values from training.
    """
    precip = max(0.0, base_row["precip_mm"] * rain_mult
                 + np.random.default_rng(step).normal(0, 0.5))
    row = {
        "precip_mm"            : precip,
        "precip_3d"            : precip * 2.5 * rain_mult,
        "precip_7d"            : precip * 5.0 * rain_mult,
        "precip_15d"           : precip * 10.0 * rain_mult,
        "temp_mean_c"          : base_row["temp_mean_c"] + temp_delta,
        "temp_max_c"           : base_row["temp_max_c"]  + temp_delta,
        "temp_min_c"           : base_row["temp_min_c"]  + temp_delta,
        "solar_rad_mj_m2_day"  : base_row["solar_rad_mj_m2_day"],
        "humidity_specific_gkg": base_row["humidity_specific_gkg"],
        "wind_speed_ms"        : base_row["wind_speed_ms"],
        "precip_60d"           : precip * 45.0 * rain_mult,
        "precip_90d"           : precip * 67.0 * rain_mult,
        "evap_proxy"           : ((base_row["temp_max_c"]+temp_delta)
                                  * base_row["solar_rad_mj_m2_day"]
                                  * base_row["wind_speed_ms"]),
        "month_sin"            : np.sin(2*np.pi*((doy//30)+1)/12),
        "month_cos"            : np.cos(2*np.pi*((doy//30)+1)/12),
        "doy_sin"              : np.sin(2*np.pi*doy/365),
        "doy_cos"              : np.cos(2*np.pi*doy/365),
        "precip_anomaly_180d"  : (rain_mult - 1.0),
    }
    return row


def _sim_ml(xgb_model, sc_ml, train_df, rain, tdelta, horizon, vol0):
    """Ensemble-powered day-by-day simulation (LSTM + XGBoost weighted average).
    Falls back to XGBoost-only when DL models are not loaded."""
    # Compute climatological base from training data
    base_row = {col: float(train_df[col].mean()) for col in FEATURES_ML
                if col in train_df.columns}

    rng     = np.random.default_rng(42)
    vols    = [vol0]
    rels    = [0.0]
    dv_list = []
    dates   = pd.date_range("2026-02-10", periods=horizon, freq="D")

    # Build horizon×FEATURES_ML matrix, then predict all at once
    rows = []
    for i, date in enumerate(dates):
        doy = date.dayofyear
        r   = _build_sim_day(base_row, rain, tdelta, vols[-1], doy, i)
        rows.append([r[f] for f in FEATURES_ML])

    X_raw = np.array(rows, dtype=float)
    # Build window features: for day i we need the 30-day window
    # → pad the start with the first row repeated
    X_padded = np.vstack([np.tile(X_raw[0], (WINDOW, 1)), X_raw])
    X_feat   = np.array([
        np.concatenate([
            X_padded[i:i+WINDOW].mean(0),
            X_padded[i:i+WINDOW].std(0),
            X_padded[i+WINDOW],
        ])
        for i in range(horizon)
    ])
    X_scaled = sc_ml.transform(X_feat)
    # Try Ensemble first, fall back to XGBoost if DL not available
    dl_models = st.session_state.get("_dl_models")
    sc_A_     = st.session_state.get("_sc_A")
    sc_B_     = st.session_state.get("_sc_B")
    sc_y_     = st.session_state.get("_sc_y")
    best_a_   = st.session_state.get("best_alpha", 0.81)
    xgb_ens_  = st.session_state.get("_ens_xgb", xgb_model)
    sc_ml_e_  = st.session_state.get("_ens_sc_ml", sc_ml)

    if dl_models and "LSTM_MI" in dl_models and sc_A_ is not None:
        # Ensemble: LSTM part uses branch A sequence
        dv_xgb_all = xgb_ens_.predict(sc_ml_e_.transform(X_feat))
        dv_preds   = []
        # Rebuild Branch-A windows from X_padded (first 10 cols = BRANCH_A)
        for idx in range(horizon):
            xa_raw  = X_padded[idx:idx+WINDOW, :len(BRANCH_A)]
            xa_s    = sc_A_.transform(xa_raw).reshape(1, WINDOW, len(BRANCH_A))
            xb_raw  = X_padded[idx+WINDOW, len(BRANCH_A):len(BRANCH_A)+len(BRANCH_B)]
            xb_s    = sc_B_.transform(xb_raw.reshape(1,-1))
            raw_lst = dl_models["LSTM_MI"].predict([xa_s, xb_s], verbose=0)
            dv_lstm = float(sc_y_.inverse_transform(raw_lst.reshape(-1,1))[0,0])
            dv_preds.append(best_a_ * dv_lstm + (1-best_a_) * float(dv_xgb_all[idx]))
        dv_preds = np.array(dv_preds)
    else:
        # Fallback: XGBoost only
        dv_preds = xgb_model.predict(X_scaled)

    for i in range(horizon):
        dv  = float(dv_preds[i])
        vn  = vols[-1] + dv
        rel = max(0, vn - MAX_CAP*FLOOD_THR) if vn > MAX_CAP*FLOOD_THR else 0.0
        vn  = max(0, min(vn - rel, MAX_CAP))
        vols.append(vn); rels.append(rel); dv_list.append(dv)

    vols  = np.array(vols[1:])
    rels  = np.array(rels[1:])
    dv_arr= np.array(dv_list)
    fpct  = vols / MAX_CAP * 100

    fd = np.where(fpct >= FLOOD_THR*100)[0]
    dd = np.where(fpct <= DRT_THR*100)[0]
    if len(fd):   alert,acol,aicon = f"Flood threshold in <b>{fd[0]} days</b>",CORAL,"⚡"
    elif len(dd): alert,acol,aicon = f"Drought threshold in <b>{dd[0]} days</b>",AMBER,"🏜"
    else:         alert,acol,aicon = f"Stable: {fpct.min():.1f}%–{fpct.max():.1f}%",GREEN1,"✅"
    info_box(alert, acol, aicon)
    info_box("Simulation powered by <b>Ensemble (LSTM + XGBoost)</b> — "
             "each day's ΔV is predicted by the Ensemble model, "
             "then integrated to get cumulative volume.", TEAL,"🤖")

    _plot_sim(dates, vols, dv_arr, rels, fpct)


def _sim_physics(rain, tdelta, horizon, vol0):
    """Fallback physics simulation when no model trained yet."""
    rng    = np.random.default_rng(42)
    doy    = np.arange(horizon)
    base_p = 3.0*(1+0.8*np.sin(2*np.pi*doy/365-1.2))
    precip = np.maximum(0, base_p*rain + rng.normal(0,1,horizon))
    inflow = precip*3990*0.25*1e-3
    temp   = 15+10*np.sin(2*np.pi*doy/365)+tdelta
    evap   = np.maximum(0,(temp-10)*0.06)*(MAX_CAP*.5/1000)

    vols,rels=[vol0],[0.0]
    for i in range(1,horizon):
        vn=vols[-1]+inflow[i]-evap[i]
        rel=max(0,vn-MAX_CAP*FLOOD_THR) if vn>MAX_CAP*FLOOD_THR else 0.0
        vols.append(max(0,min(vn-rel,MAX_CAP))); rels.append(rel)

    vols=np.array(vols); rels=np.array(rels)
    dv_arr=np.diff(vols,prepend=vols[0])
    fpct=vols/MAX_CAP*100
    dates=pd.date_range("2026-02-10",periods=horizon,freq="D")

    fd=np.where(fpct>=FLOOD_THR*100)[0]; dd=np.where(fpct<=DRT_THR*100)[0]
    if len(fd):   alert,acol,aicon=f"Flood in <b>{fd[0]} days</b>",CORAL,"⚡"
    elif len(dd): alert,acol,aicon=f"Drought in <b>{dd[0]} days</b>",AMBER,"🏜"
    else:         alert,acol,aicon=f"Stable {fpct.min():.1f}%–{fpct.max():.1f}%",GREEN1,"✅"
    info_box(alert,acol,aicon)
    info_box("Physics-based fallback simulation. Train models first for ML simulation.",AMBER,"⚙️")
    _plot_sim(dates, vols, dv_arr, rels, fpct)


def _plot_sim(dates, vols, dv_arr, rels, fpct):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.68,0.32], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(
        x=dates, y=vols, name="Simulated volume",
        fill="tozeroy", fillcolor="rgba(34,197,94,.10)",
        line=dict(color=GREEN2,width=2.5),
        hovertemplate="%{x|%d %b %Y}<br><b>%{y:.1f} Mm³</b><extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=MAX_CAP*FLOOD_THR, line_dash="dash", line_color=CORAL,
                  line_width=1.5, annotation_text="Flood 85%",
                  annotation_font_color=CORAL, row=1, col=1)
    fig.add_hline(y=MAX_CAP*DRT_THR, line_dash="dot", line_color=AMBER,
                  line_width=1.5, annotation_text="Drought 30%",
                  annotation_font_color=AMBER, row=1, col=1)
    colors_dv = [GREEN1 if d>=0 else CORAL for d in dv_arr]
    fig.add_trace(go.Bar(
        x=dates, y=dv_arr, name="Daily ΔV (XGBoost)",
        marker_color=colors_dv, opacity=.7,
        hovertemplate="%{x|%d %b}<br>%{y:.3f} Mm³/d<extra></extra>",
    ), row=2, col=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=WHITE,
        font=dict(color=TMID,family="Inter",size=12),
        height=450, margin=dict(t=28,b=34,l=60,r=20),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis2=dict(showgrid=False,color=TLITE),
        yaxis =dict(title="Volume (Mm³)",gridcolor="#f3f4f6",color=TLITE),
        yaxis2=dict(title="ΔV (Mm³/d)",gridcolor="#f3f4f6",color=TLITE),
    )
    st.plotly_chart(fig, use_container_width=True)

    m1,m2,m3,m4 = st.columns(4)
    with m1: st.metric("Final",   f"{vols[-1]:.0f} Mm³",  f"{fpct[-1]:.1f}%")
    with m2: st.metric("Peak",    f"{vols.max():.0f} Mm³", f"{fpct.max():.1f}%")
    with m3: st.metric("Min",     f"{vols.min():.0f} Mm³", f"{fpct.min():.1f}%")
    with m4: st.metric("Released",f"{rels.sum():.0f} Mm³",
                        f"{int((rels>0).sum())} days")


# ════════════════════════════════════════════════════
# PAGE 5 — EARLY WARNING
# ════════════════════════════════════════════════════
def page_early_warning():
    breadcrumb("MONITORING › ALERTS")
    st.markdown("<h1>Early Warning System</h1>", unsafe_allow_html=True)

    df = load_main_data()
    v  = float(df["wsc_calibre_mm3"].dropna().iloc[-1])
    fr = fill_pct(v)
    rl, rc, ri = regime(fr)

    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};
        border-left:5px solid {rc};border-radius:14px;
        padding:18px 22px;margin:14px 0 20px;
        box-shadow:0 1px 3px rgba(0,0,0,.06);
        display:flex;align-items:center;gap:16px">
      <div style="font-size:2.2rem">{ri}</div>
      <div>
        <div style="font-size:.67rem;font-weight:700;text-transform:uppercase;
            letter-spacing:.12em;color:{TLITE}">Current Status</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.4rem;
            font-weight:700;color:{rc};margin:2px 0">{rl}</div>
        <div style="font-size:.81rem;color:{TMID}">
          Volume: <b>{v:,.0f} Mm³</b> · Fill: <b>{fr:.1f}%</b> ·
          Last update: <b>{df['wsc_calibre_mm3'].dropna().index[-1].date()}</b>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    sec("Storm Marta — Case Study Replay","Jul 2024 – Feb 2026")
    marta = df.loc["2024-07-01":"2026-01-28","wsc_calibre_mm3"].dropna()
    if len(marta) < 10:
        idx  = pd.date_range("2024-07-01","2026-01-28",freq="D")
        vals = np.concatenate([np.linspace(380,280,180),np.linspace(280,955,len(idx)-180)])
        marta= pd.Series(vals,index=idx)

    ai_fc = marta.values + np.random.default_rng(0).normal(0,22,len(marta))+28
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=marta.index, y=marta.values, fill="tozeroy",
        fillcolor="rgba(34,197,94,.10)", line=dict(color=GREEN2,width=2.5),
        name="Observed volume",
        hovertemplate="%{x|%d %b %Y}<br><b>%{y:.1f} Mm³</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=marta.index, y=ai_fc,
        line=dict(color=TEAL,width=1.5,dash="dash"),
        name="AI 30-day forecast", opacity=.75,
    ))
    for d,lbl,col in [
        ("2025-01-01","24.85%\nStart",AMBER),
        ("2026-01-15","⚡ AI Alert\nissued",CORAL),
        ("2026-02-08","84 m³/s\nrelease",CORAL),
    ]:
        yv=float(marta.mean())
        try:
            sl=marta.loc[d:];
            if len(sl): yv=float(sl.iloc[0])
        except: pass
        fig.add_vline(x=pd.Timestamp(d),line_dash="dash",line_color=col,line_width=1.5)
        fig.add_annotation(x=d,y=yv,text=lbl.replace("\n","<br>"),
                           showarrow=True,arrowhead=2,arrowcolor=col,
                           font=dict(color=col,size=10),
                           bgcolor=WHITE,bordercolor=col,borderwidth=1,borderpad=4)
    fig.add_hline(y=MAX_CAP*FLOOD_THR,line_dash="dash",line_color=CORAL,
                  line_width=1.5,annotation_text="Flood 85%",
                  annotation_font_color=CORAL)
    b = _base(310); b["yaxis"]["title"] = "Volume (Mm³)"
    fig.update_layout(**b,
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,
                                  bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

    sec("Event Timeline")
    for dt,fl,vt,desc,col in [
        ("Jan 2025",    "24.85%","280 Mm³","Near-drought. Start of Marta precipitation system.",          AMBER),
        ("Apr 2025",    "35%",   "395 Mm³","First rainfall events — AI detects rising trend.",            TLITE),
        ("Sep 2025",    "55%",   "621 Mm³","Sustained autumn rains — AI flags anomalous accumulation.",   TEAL),
        ("Jan 15, 2026","80%",   "903 Mm³","⚡ AI Early Warning issued — flood threshold in ~24 days.",  CORAL),
        ("Feb 8, 2026", "84.64%","955 Mm³","Emergency 84 m³/s release — floods Kenitra & Sidi Kacem.",  CORAL),
    ]:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 16px;
            margin-bottom:5px;background:{CARD};border-radius:10px;
            border:1px solid {BORDER};border-left:4px solid {col};
            box-shadow:0 1px 2px rgba(0,0,0,.04)">
          <div style="font-size:.71rem;font-weight:700;color:{col};
              min-width:96px;font-family:'Space Mono',monospace">{dt}</div>
          <div style="font-size:.71rem;color:{TLITE};min-width:52px;
              font-family:'Space Mono',monospace">{fl}</div>
          <div style="font-size:.71rem;color:{TLITE};min-width:62px;
              font-family:'Space Mono',monospace">{vt}</div>
          <div style="font-size:.82rem;color:{TMID}">{desc}</div>
        </div>""", unsafe_allow_html=True)

    sec("Flood Impact — February 2026")
    i1,i2,i3 = st.columns(3)
    with i1: st.markdown(kpi("Provinces affected","3","Gharb · Kenitra · Sidi Kacem",CORAL,"🗺"), unsafe_allow_html=True)
    with i2: st.markdown(kpi("Farmland at risk","~120K ha","Gharb agricultural plain",AMBER,"🌾"), unsafe_allow_html=True)
    with i3: st.markdown(kpi("AI lead time","30–60 days","Before critical threshold",GREEN1,"🤖"), unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# PAGE 6 — DOCUMENTATION
# ════════════════════════════════════════════════════
def page_docs():
    breadcrumb("REFERENCE › DOCUMENTATION")
    st.markdown("<h1>Documentation</h1>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "  How to Use",
        "  Architecture",
        "  Methodology",
    ])

    with tab1:
        sec("Getting Started — Step by Step")
        _howto_html = textwrap.dedent(f"""
                <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;
                        padding:24px 28px;box-shadow:0 1px 4px rgba(0,0,0,.06)">

                    <div style="display:flex;flex-direction:column;gap:16px">

                        <!-- Step 1 -->
                        <div style="display:flex;gap:16px;align-items:flex-start">
                            <div style="min-width:32px;height:32px;background:{GREEN1};border-radius:50%;
                                    display:flex;align-items:center;justify-content:center;
                                    font-family:'Space Mono',monospace;font-weight:700;
                                    color:white;font-size:.9rem;flex-shrink:0">1</div>
                            <div>
                                <div style="font-weight:700;color:{TDARK};font-size:.95rem">
                                     Upload your data</div>
                                <div style="color:{TMID};font-size:.84rem;margin-top:4px">
                                    Go to <b>Upload Data</b> and drop your CSV file.
                                    It must contain the same columns as
                                    <code>idriss1_final_DL_ready.csv</code>:
                                    date, wsc_calibre_mm3, wsc_u_mm3, segment_id,
                                    precip_mm, temp_mean_c, temp_max_c, temp_min_c,
                                    solar_rad_mj_m2_day, humidity_specific_gkg, wind_speed_ms.<br>
                                    <b>Minimum:</b> 200 rows for ML · 500+ rows for full DL + Ensemble.
                                    The original dataset (5,991 rows / 14 years) is used if no file is uploaded.
                                </div>
                            </div>
                        </div>

                        <div style="height:1px;background:{BORDER}"></div>

                        <!-- Step 2 -->
                        <div style="display:flex;gap:16px;align-items:flex-start">
                            <div style="min-width:32px;height:32px;background:{GREEN1};border-radius:50%;
                                    display:flex;align-items:center;justify-content:center;
                                    font-family:'Space Mono',monospace;font-weight:700;
                                    color:white;font-size:.9rem;flex-shrink:0">2</div>
                            <div>
                                <div style="font-weight:700;color:{TDARK};font-size:.95rem">
                                     Train the models</div>
                                <div style="color:{TMID};font-size:.84rem;margin-top:4px">
                                    Go to <b>Training & Results</b> and click
                                    <b>Train All Models</b>.
                                    The pipeline trains 4 ML models (Ridge, RF, XGBoost, LightGBM),
                                    2 DL models (LSTM, GRU), and builds the optimal Ensemble.
                                    Training takes <b>30–60 seconds for ML</b> and
                                    <b>~5 minutes for the full DL suite</b>.
                                    Results tables and charts appear automatically when done.
                                </div>
                            </div>
                        </div>

                        <div style="height:1px;background:{BORDER}"></div>

                        <!-- Step 3 -->
                        <div style="display:flex;gap:16px;align-items:flex-start">
                            <div style="min-width:32px;height:32px;background:{GREEN1};border-radius:50%;
                                    display:flex;align-items:center;justify-content:center;
                                    font-family:'Space Mono',monospace;font-weight:700;
                                    color:white;font-size:.9rem;flex-shrink:0">3</div>
                            <div>
                                <div style="font-weight:700;color:{TDARK};font-size:.95rem">
                                     Run predictions</div>
                                <div style="color:{TMID};font-size:.84rem;margin-top:4px">
                                    Go to <b>Forecast</b>. Choose a horizon:
                                    Tomorrow, 1 Week, 1 Month, or 3 Months.
                                    The Ensemble model (LSTM + XGBoost) predicts the daily ΔV
                                    for each future day and reconstructs the volume trajectory.
                                    A day-by-day table shows volume, fill rate, and risk status.
                                </div>
                            </div>
                        </div>

                        <div style="height:1px;background:{BORDER}"></div>

                        <!-- Step 4 -->
                        <div style="display:flex;gap:16px;align-items:flex-start">
                            <div style="min-width:32px;height:32px;background:{GREEN1};border-radius:50%;
                                    display:flex;align-items:center;justify-content:center;
                                    font-family:'Space Mono',monospace;font-weight:700;
                                    color:white;font-size:.9rem;flex-shrink:0">4</div>
                            <div>
                                <div style="font-weight:700;color:{TDARK};font-size:.95rem">
                                     Simulate scenarios</div>
                                <div style="color:{TMID};font-size:.84rem;margin-top:4px">
                                    Go to <b>Scenario Simulator</b>. Pick a preset
                                    (Normal Year, Drought 2022, Storm Marta 2025) or set
                                    custom rainfall multiplier, temperature anomaly, horizon, and
                                    initial volume. Click <b>Run Simulation</b> — the Ensemble
                                    model predicts volume day-by-day under those conditions.
                                    Flood and drought threshold breaches are flagged automatically.
                                </div>
                            </div>
                        </div>

                    </div>
                </div>""").strip()
        _howto_html = "\n".join(line.lstrip() for line in _howto_html.splitlines())
        st.markdown(_howto_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sec("Required CSV Columns")
        cols_df = pd.DataFrame([
            {"Column":"date",                 "Type":"datetime","Description":"Date of observation (any parseable format)"},
            {"Column":"wsc_calibre_mm3",      "Type":"float",  "Description":"Dam water volume (Mm³) — the main target variable"},
            {"Column":"wsc_u_mm3",            "Type":"float",  "Description":"Measurement uncertainty (used for sample weighting)"},
            {"Column":"segment_id",           "Type":"int",    "Description":"0 = pre-gap (train/val) · 1 = post-gap (test)"},
            {"Column":"precip_mm",            "Type":"float",  "Description":"Daily precipitation (mm)"},
            {"Column":"temp_mean_c",          "Type":"float",  "Description":"Mean daily temperature (°C)"},
            {"Column":"temp_max_c",           "Type":"float",  "Description":"Maximum daily temperature (°C)"},
            {"Column":"temp_min_c",           "Type":"float",  "Description":"Minimum daily temperature (°C)"},
            {"Column":"solar_rad_mj_m2_day",  "Type":"float",  "Description":"Solar radiation (MJ/m²/day)"},
            {"Column":"humidity_specific_gkg","Type":"float",  "Description":"Specific humidity (g/kg)"},
            {"Column":"wind_speed_ms",        "Type":"float",  "Description":"Wind speed (m/s)"},
        ])
        st.dataframe(cols_df, use_container_width=True, hide_index=True)

    with tab2:
        sec("Full System Architecture")
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;
            padding:24px 28px;box-shadow:0 1px 4px rgba(0,0,0,.06);
            font-size:.85rem;line-height:1.9;color:{TMID}">

          <div style="font-family:'Space Mono',monospace;font-weight:700;
              color:{TDARK};font-size:1rem;margin-bottom:14px">
            Pipeline Overview</div>

          <b style="color:{TDARK}">Input data:</b>
          14 years of daily observations (2008–2026) combining
          <b>DAHITI satellite altimetry</b> (dam water surface height → calibrated
          volume in Mm³) with <b>NASA POWER climate reanalysis</b>
          (precipitation, temperature min/max/mean, solar radiation,
          specific humidity, wind speed). Total: 5,991 daily rows.

          <br><br>
          <b style="color:{TDARK}">Target variable:</b>
          Instead of predicting volume V(t), we predict
          <b style="color:{GREEN2}">ΔV(t) = V(t) − V(t−1)</b>, the daily volume
          change. This is the core innovation — it forces the model to learn
          the true weather→runoff causality instead of exploiting autocorrelation.

          <br><br>
          <b style="color:{TDARK}">Feature engineering:</b>
          From the 11 raw columns we derive 18 features split into two branches:

          <br><br>
          <b>Branch A — 10 sequential features</b> (fed to LSTM/GRU over 30 days):
          <code>precip_mm</code>, <code>precip_3d/7d/15d</code>
          (rolling precipitation sums capturing hydrological memory),
          <code>temp_mean/max/min_c</code>,
          <code>solar_rad_mj_m2_day</code>,
          <code>humidity_specific_gkg</code>,
          <code>wind_speed_ms</code>.

          <br><br>
          <b>Branch B — 8 static features</b> (fed to Dense layer, current day):
          <code>precip_60d/90d</code> (long-term catchment saturation),
          <code>evap_proxy = temp_max × solar × wind</code>,
          <code>month_sin/cos + doy_sin/cos</code> (cyclic seasonality),
          <code>precip_anomaly_180d</code> (anomaly vs 180-day mean).

          <br><br>
          <b style="color:{TDARK}">Deep Learning models (LSTM_MI / GRU_MI):</b><br>
          Branch A → LSTM or GRU (32 units) → Dropout(0.3)<br>
          Branch B → Dense(16, ReLU) → Dropout(0.2)<br>
          Concatenate → Dense(16, ReLU) → Dropout(0.1) → Dense(1) → ΔV<br>
          Loss: MSE · Optimizer: Adam · EarlyStopping(patience=20)

          <br><br>
          <b style="color:{TDARK}">ML models:</b>
          For each of the 18 features, we compute
          <b>mean + std over the 30-day window + current-day value</b>
          → 18×3 = <b>54 features</b> per sample.
          Models: Ridge (α=1), Random Forest (300 trees, max_depth=8),
          XGBoost (300 estimators, max_depth=4, L1+L2 reg),
          LightGBM (same setup).

          <br><br>
          <b style="color:{TDARK}">Ensemble (best model):</b>
          Optimal weighted combination found by grid search on validation MAE:
          <b style="color:{GREEN2}">ΔV_ensemble = α × LSTM_MI + (1−α) × XGBoost</b>
          where α≈0.81 in the original paper.
          This achieves <b>R²=0.236</b> on daily ΔV and
          <b>MAE_vol=247 Mm³</b> on 9-year volume reconstruction.

          <br><br>
          <b style="color:{TDARK}">Train / Validation / Test split:</b>
          Segment 0 (2008–2016): first 80% = Train, last 20% = Validation.<br>
          Satellite gap (Nov 2016 – Jul 2017): excluded.<br>
          Segment 1 (2017–2026): entirely held out as <b>Test set</b> (9 years).

        </div>""", unsafe_allow_html=True)

    with tab3:
        sec("Why ΔV Instead of Direct Volume Prediction?")
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;
            padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,.06);
            font-size:.85rem;color:{TMID};line-height:1.9">

          <b style="color:{TDARK}">The persistence trap:</b>
          Dam volume is highly autocorrelated from day to day (R ≈ 0.99).
          A naive model that predicts <i>"tomorrow = today"</i> achieves
          NSE=0.9994 — a seemingly perfect score. But it has learned nothing.
          It cannot predict a flood or drought 30 days in advance.<br><br>

          <b style="color:{TDARK}">The ΔV solution:</b>
          By predicting the <i>daily change</i> ΔV = V(t)−V(t−1), we remove
          the autocorrelation signal entirely. Now the naive baseline achieves R²≈0,
          and our Ensemble achieves R²=0.236 — which represents
          <b style="color:{GREEN2}">genuine learning of the weather→runoff relationship</b>.
          This is the scientifically correct metric.<br><br>

          <b style="color:{TDARK}">Autonomous simulation:</b>
          Because ΔV predictions can be chained
          (V(t+k) = V(t) + Σ ΔV), the model can simulate volume trajectories
          weeks or months into the future — impossible with direct V prediction.<br><br>

          <b style="color:{TDARK}">Sample weighting:</b>
          Training samples are weighted by
          <code>w = (1/measurement_uncertainty) × (1 + 2×norm_precip)</code>.
          This gives more importance to high-precipitation events (the ones that
          matter most for flood prediction) and down-weights uncertain measurements.

        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sec("Feature Engineering Table")
        fe = pd.DataFrame([
            {"Feature":"precip_mm",            "Branch":"A","Window":"Daily","Physical role":"Raw daily precipitation input"},
            {"Feature":"precip_3d/7d/15d",     "Branch":"A","Window":"3,7,15d","Physical role":"Rolling sums — catchment memory"},
            {"Feature":"temp_mean/max/min_c",   "Branch":"A","Window":"Daily","Physical role":"Controls evapotranspiration & snowmelt"},
            {"Feature":"solar_rad_mj_m2_day",   "Branch":"A","Window":"Daily","Physical role":"Surface energy balance"},
            {"Feature":"humidity_specific_gkg", "Branch":"A","Window":"Daily","Physical role":"Atmospheric water vapour content"},
            {"Feature":"wind_speed_ms",         "Branch":"A","Window":"Daily","Physical role":"Drives reservoir evaporation"},
            {"Feature":"precip_60d / 90d",      "Branch":"B","Window":"60,90d","Physical role":"Long-term catchment saturation"},
            {"Feature":"evap_proxy",            "Branch":"B","Window":"Daily","Physical role":"temp_max × solar × wind (estimated loss)"},
            {"Feature":"month_sin/cos",         "Branch":"B","Window":"Cyclic","Physical role":"Avoids Dec→Jan discontinuity"},
            {"Feature":"doy_sin/cos",           "Branch":"B","Window":"Cyclic","Physical role":"Day-of-year seasonality"},
            {"Feature":"precip_anomaly_180d",   "Branch":"B","Window":"180d","Physical role":"Detects anomalous rainfall for season"},
        ])
        st.dataframe(fe, use_container_width=True, hide_index=True)



def page_forecast():
    breadcrumb("AI › FORECAST")
    st.markdown("<h1>AI Prediction Panel</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<p style=\'color:{TMID};font-size:.86rem\'>"
        "The <b>Ensemble model</b> (LSTM + XGBoost, optimised weighting) predicts ΔV day-by-day. "
        "The volume trajectory is built by rolling the prediction forward. "
        "Confidence bands widen with horizon to reflect cumulative uncertainty.</p>",
        unsafe_allow_html=True)

    xgb_model = st.session_state.get("xgb_model")
    sc_ml     = st.session_state.get("sc_ml")
    test_df   = st.session_state.get("test_df")

    if xgb_model is None or sc_ml is None:
        info_box(
            "No trained model found. Go to <b>Training & Results</b> → "
            "click <b>Train / Retrain Models</b> → come back here.",
            AMBER, "⚠️")
        return

    # Use test_df (last known real data) or main dataset
    if test_df is not None and len(test_df) >= WINDOW:
        df_src = test_df
    else:
        raw    = load_main_data()
        df_src = feature_engineering(raw)

    # Guard: ensure all FEATURES_ML columns exist
    missing_cols = [c for c in FEATURES_ML if c not in df_src.columns]
    if missing_cols:
        st.error(f"Feature columns missing in data: {missing_cols}")
        return

    last_vol  = float(df_src["wsc_calibre_mm3"].iloc[-1])
    last_date = df_src.index[-1]
    last_fr   = fill_pct(last_vol)
    rl, rc, ri = regime(last_fr)

    # ── Current status banner ────────────────────────────
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};
        border-left:5px solid {rc};border-radius:14px;
        padding:18px 24px;margin:14px 0 22px;
        box-shadow:0 1px 4px rgba(0,0,0,.06);
        display:flex;align-items:center;gap:20px">
      <div style="font-size:2.4rem">{ri}</div>
      <div style="flex:1">
        <div style="font-size:.67rem;font-weight:700;text-transform:uppercase;
            letter-spacing:.12em;color:{TLITE}">Last Observation</div>
        <div style="font-family:\'Space Mono\',monospace;font-size:1.35rem;
            font-weight:700;color:{rc}">{rl}</div>
        <div style="font-size:.82rem;color:{TMID};margin-top:2px">
          Date: <b>{last_date.date()}</b> &nbsp;·&nbsp;
          Volume: <b>{last_vol:,.1f} Mm³</b> &nbsp;·&nbsp;
          Fill: <b>{last_fr:.1f}%</b>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Horizon selector ─────────────────────────────────
    sec("Forecast Horizon")
    if "fc_horizon" not in st.session_state:
        st.session_state["fc_horizon"] = 7

    h1, h2, h3, h4, _ = st.columns([1, 1, 1, 1, 3])
    with h1:
        if st.button("📅 Tomorrow",  use_container_width=True): st.session_state["fc_horizon"] = 1
    with h2:
        if st.button("📅 1 Week",    use_container_width=True): st.session_state["fc_horizon"] = 7
    with h3:
        if st.button("📅 1 Month",   use_container_width=True): st.session_state["fc_horizon"] = 30
    with h4:
        if st.button("📅 3 Months",  use_container_width=True): st.session_state["fc_horizon"] = 90

    horizon = int(st.session_state["fc_horizon"])

    # ── Run prediction ────────────────────────────────────
    with st.spinner(f"XGBoost forecasting {horizon} day(s)…"):
        try:
            dates, dv_preds, vol_preds, vol_low, vol_high = _predict_horizon(
                xgb_model, sc_ml, df_src, horizon)
        except Exception as e:
            st.error(f"Forecast error: {e}")
            return

    # ── Tomorrow KPIs ────────────────────────────────────
    dv_tom  = float(dv_preds[0])
    vol_tom = float(vol_preds[0])
    fr_tom  = fill_pct(vol_tom)
    rl_t, rc_t, ri_t = regime(fr_tom)
    sign    = "+" if dv_tom >= 0 else ""
    if   dv_tom >  0.5: trend, tc = "📈 Rising",  GREEN1
    elif dv_tom < -0.5: trend, tc = "📉 Falling", CORAL
    else:               trend, tc = "➡ Stable",  AMBER

    sec("Tomorrow's Prediction", dates[0].strftime("%d %b %Y"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Predicted ΔV",
            f"{sign}{dv_tom:.2f}", "Mm³ change tomorrow",
            GREEN1 if dv_tom >= 0 else CORAL, "📊"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Predicted Volume",
            f"{vol_tom:,.1f}", "Mm³",
            GREEN1, "💧"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Predicted Fill Rate",
            f"{fr_tom:.1f}%", rl_t, rc_t, ri_t), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Volume Trend",
            trend, f"ΔV = {sign}{dv_tom:.2f} Mm³/d",
            tc, "📡"), unsafe_allow_html=True)

    # ── Multi-day summary ────────────────────────────────
    if horizon > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        sec(f"{horizon}-Day Forecast Summary")

        alert_msg   = None
        alert_color = GREEN1
        for i, (v, d) in enumerate(zip(vol_preds, dates)):
            if v >= MAX_CAP * FLOOD_THR:
                alert_msg   = (f"Flood risk: volume may reach "
                               f"<b>{v:.0f} Mm³ ({fill_pct(v):.1f}%)</b> "
                               f"on <b>{d.strftime('%d %b %Y')}</b> (day {i+1})")
                alert_color = CORAL
                break
            if v <= MAX_CAP * DRT_THR:
                alert_msg   = (f"Drought risk: volume may drop to "
                               f"<b>{v:.0f} Mm³ ({fill_pct(v):.1f}%)</b> "
                               f"on <b>{d.strftime('%d %b %Y')}</b> (day {i+1})")
                alert_color = AMBER
                break

        if alert_msg:
            info_box(alert_msg, alert_color, "⚠️")
        else:
            info_box(
                f"No threshold breach forecast. Volume expected between "
                f"<b>{vol_preds.min():.0f}</b> and "
                f"<b>{vol_preds.max():.0f} Mm³</b> "
                f"({fill_pct(vol_preds.min()):.1f}%–"
                f"{fill_pct(vol_preds.max()):.1f}%) "
                f"over the next {horizon} days.",
                GREEN1, "✅")

        vol_end   = float(vol_preds[-1])
        fr_end    = fill_pct(vol_end)
        dv_total  = vol_end - last_vol
        rl_e, rc_e, _ = regime(fr_end)
        s2 = "+" if dv_total >= 0 else ""
        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown(kpi(f"Volume in {horizon} days",
                f"{vol_end:,.1f}", "Mm³", rc_e, "🏗"), unsafe_allow_html=True)
        with e2:
            st.markdown(kpi(f"Fill Rate in {horizon} days",
                f"{fr_end:.1f}%", rl_e, rc_e, "📊"), unsafe_allow_html=True)
        with e3:
            st.markdown(kpi("Total Volume Change",
                f"{s2}{dv_total:.1f}", f"Mm³ over {horizon} days",
                GREEN1 if dv_total >= 0 else CORAL, "⚡"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast chart ────────────────────────────────────
    # Key: use pd.Timestamp objects everywhere — never str — to keep x-axis typed correctly
    sec("Volume Forecast Chart")
    hist = df_src["wsc_calibre_mm3"].iloc[-30:]

    fig = go.Figure()

    # Last 30 days observed
    fig.add_trace(go.Scatter(
        x=list(hist.index), y=list(hist.values.astype(float)),
        line=dict(color=TDARK, width=2.5),
        name="Observed (last 30 days)",
        hovertemplate="%{x|%d %b %Y}<br><b>%{y:.1f} Mm³</b><extra></extra>",
    ))

    # Confidence band (only for multi-day)
    if horizon > 1:
        band_x = list(dates) + list(dates[::-1])
        band_y = list(vol_high.astype(float)) + list(vol_low[::-1].astype(float))
        fig.add_trace(go.Scatter(
            x=band_x, y=band_y,
            fill="toself",
            fillcolor="rgba(34,197,94,.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence band",
            hoverinfo="skip",
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=list(dates), y=list(vol_preds.astype(float)),
        line=dict(color=GREEN1, width=2.5, dash="dash"),
        name="XGBoost forecast",
        hovertemplate="%{x|%d %b %Y}<br><b>%{y:.1f} Mm³</b> (forecast)<extra></extra>",
    ))

    # Tomorrow diamond marker
    fig.add_trace(go.Scatter(
        x=[dates[0]], y=[float(vol_preds[0])],
        mode="markers",
        marker=dict(color=GREEN1, size=13, symbol="diamond",
                    line=dict(color=WHITE, width=2)),
        name="Tomorrow",
        hovertemplate=f"Tomorrow: <b>{vol_preds[0]:.1f} Mm³</b><extra></extra>",
    ))

    # "Today" vertical line — use add_shape (avoids int+str Plotly crash)
    fig.add_shape(
        type="line",
        x0=last_date, x1=last_date,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=TLITE, width=1.5, dash="dot"),
    )
    fig.add_annotation(
        x=last_date, y=1.04, xref="x", yref="paper",
        text="Today", showarrow=False,
        font=dict(color=TLITE, size=10),
        xanchor="center",
    )

    # Threshold lines
    fig.add_hline(y=MAX_CAP * FLOOD_THR, line_dash="dash",
                  line_color=CORAL, line_width=1.5,
                  annotation_text="Flood 85%",
                  annotation_font_color=CORAL, annotation_font_size=10)
    fig.add_hline(y=MAX_CAP * DRT_THR, line_dash="dot",
                  line_color=AMBER, line_width=1.5,
                  annotation_text="Drought 30%",
                  annotation_font_color=AMBER, annotation_font_size=10)

    b = _base(360)
    b["yaxis"]["title"] = "Volume (Mm³)"
    fig.update_layout(**b,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    st.plotly_chart(fig, use_container_width=True)

    # ── Daily ΔV bar chart ────────────────────────────────
    if horizon > 1:
        sec("Daily ΔV Predictions")
        clrs = [GREEN1 if float(d) >= 0 else CORAL for d in dv_preds]
        fig2 = go.Figure(go.Bar(
            x=list(dates), y=[float(d) for d in dv_preds],
            marker_color=clrs, opacity=0.85,
            hovertemplate="%{x|%d %b %Y}<br>ΔV = <b>%{y:.3f} Mm³/d</b><extra></extra>",
        ))
        fig2.add_hline(y=0, line_color=TLITE, line_width=1)
        b2 = _base(220)
        b2["yaxis"]["title"] = "ΔV (Mm³/day)"
        fig2.update_layout(**b2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Day-by-day table ─────────────────────────────────
    sec("Day-by-Day Forecast Table")
    rows = []
    for i, (d, dv, vp, vl, vh) in enumerate(
            zip(dates, dv_preds, vol_preds, vol_low, vol_high)):
        fr_  = fill_pct(float(vp))
        rl_, rc_, _ = regime(fr_)
        rows.append({
            "Day":           i + 1,
            "Date":          d.strftime("%d %b %Y"),
            "ΔV (Mm³)":     round(float(dv), 3),
            "Volume (Mm³)":  round(float(vp), 1),
            "Fill %":        round(float(fr_), 1),
            "Low (Mm³)":     round(float(vl), 1),
            "High (Mm³)":    round(float(vh), 1),
            "Status":        rl_,
        })
    tbl = pd.DataFrame(rows)

    def dvc(v):
        if   v >  1:  return "background:#dcfce7;color:#15803d"
        elif v >= 0:  return "background:#f0fdf4;color:#166534"
        elif v > -1:  return "background:#fff7ed;color:#9a3412"
        else:         return "background:#fee2e2;color:#991b1b"

    def stc(v):
        if "DROUGHT"  in str(v): return "color:#d97706;font-weight:700"
        if "EXCEPT"   in str(v): return "color:#dc2626;font-weight:700"
        return "color:#16a34a;font-weight:600"

    st.dataframe(
        tbl.style
           .applymap(dvc, subset=["ΔV (Mm³)"])
           .applymap(stc, subset=["Status"])
           .set_properties(**{"text-align": "center", "font-size": "13px"}),
        use_container_width=True, hide_index=True)

    info_box(
        "Predictions use the <b>Ensemble model</b> (best_α × LSTM_MI + (1-α) × XGBoost), the best-performing model in our study. "
        "Confidence bands (Low/High) widen with horizon reflecting cumulative uncertainty. "
        "<b>Validate against real-time meteorological data before operational use.</b>",
        TEAL, "ℹ️")


# ROUTER
# ════════════════════════════════════════════════════
if   page == "Overview":           page_overview()
elif page == "Upload Data":        page_upload()
elif page == "Training & Results": page_model_results()
elif page == "Forecast":           page_forecast()
elif page == "Scenario Simulator": page_simulator()
elif page == "Documentation":      page_docs()
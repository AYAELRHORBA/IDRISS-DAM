# 💧 Idriss I Dam — AI Water Intelligence Dashboard

> **AI Systems for Optimal Control and Management of Climate Impact on Water Resources**  
> Case Study: Idriss I Dam, Sebou Basin, Morocco  
> Authors: EL MANSOURI Aya · EL RHORBA Aya · 

---

## Table of Contents

1. [What This App Does](#what-this-app-does)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Running the App](#running-the-app)
6. [File Descriptions](#file-descriptions)
7. [How to Use — Step by Step](#how-to-use--step-by-step)
8. [Data Format](#data-format)
9. [Model Files](#model-files)
10. [Deploying Online (Streamlit Cloud)](#deploying-online-streamlit-cloud)
11. [Troubleshooting](#troubleshooting)

---

## What This App Does

This is a 6-page Streamlit dashboard that:

- **Monitors** the Idriss I Dam in real time using satellite altimetry data (DAHITI) merged with NASA POWER climate data
- **Trains** 6 AI models (Ridge, Random Forest, XGBoost, LightGBM, LSTM, GRU) + an Ensemble on your data
- **Forecasts** dam volume 1 day to 3 months ahead using the Ensemble model
- **Simulates** climate scenarios (Normal Year / Drought 2022 / Storm Marta 2025) with customisable parameters
- **Warns** about flood and drought thresholds in advance

The core scientific innovation is predicting **daily volume change (ΔV)** instead of volume directly, which avoids the "persistence trap" (NSE = 0.9994 for a model that just copies yesterday's value).

---


```


---

## Requirements

- **Python:** 3.9, 3.10, or 3.11 (recommended: 3.11)
- **OS:** Windows, macOS, or Linux
- **RAM:** At least 4 GB (8 GB recommended for DL training)
- **Disk:** At least 500 MB free

---

## Installation

### Step 1 — Clone or download the project

```bash
# If you have git:
git clone https://github.com/your-username/idriss-dam-ai.git
cd idriss-dam-ai

# Or just download the ZIP and extract it, then open a terminal in the folder
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv311
venv311\Scripts\activate

# macOS / Linux
python3 -m venv venv311
source venv311/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

**Or install manually:**

```bash
pip install streamlit==1.35.0
pip install pandas numpy scikit-learn
pip install xgboost lightgbm
pip install plotly
pip install tensorflow==2.15.0
pip install joblib
```

> **TensorFlow note:** On Apple Silicon (M1/M2/M3 Mac), use `tensorflow-macos` instead:
> ```bash
> pip install tensorflow-macos
> ```

---

## Running the App

Make sure your virtual environment is activated and you are in the project folder, then:

```bash
streamlit run app.py
```

Your browser will automatically open at `http://localhost:8501`

To stop the app press `Ctrl+C` in the terminal.

---



"""Master configuration for basket dispersion trading project."""

import numpy as np

# ASSET UNIVERSE

SX5E_TICKER = "^STOXX50E"

TOP_20_CONSTITUENTS = [
    "SAP",
    "ASML.AS",
    "MC.PA",
    "ALV.DE",
    "SIE.DE",
    "BMW.DE",
    "MBG.DE",
    "DB1.DE",
    "BAYN.DE",
    "BNP.PA",
    "SAN.PA",
    "OR.PA",
    "NOKIA.HE",
    "TEF.MC",
    "KER.PA",
    "AI.PA",
    "GLE.PA",
    "INGA.AS",
    "ENGI.PA",
    "ISP.MI",
]

# DATA WINDOWS

BACKTEST_START = "2018-01-01"
BACKTEST_END = "2025-12-31"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2024-09-30"
TEST_START = "2024-10-01"
TEST_END = BACKTEST_END

# OPTION PARAMETERS

MATURITY = 0.25
MONEYNESS_GRID = [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20]
REALIZED_VOL_WINDOW = 30
SKEW_PARAM_ALPHA = -0.15
SMILE_PARAM_BETA = 0.05

# MARKET PARAMETERS

EURIBOR_3M = 0.035
DIV_YIELD_INDEX = 0.025
DIV_YIELD_STOCK = 0.020

# FEATURE ENGINEERING

FEATURE_DIM = 274
SCALER_EPSILON = 1e-8

# PATHS

DATA_DIR = "output/datasets"
MODEL_DIR = "output/models"
BACKTEST_DIR = "output/backtest_results"

# UTILITY FUNCTIONS

def get_all_tickers():
    return [SX5E_TICKER] + TOP_20_CONSTITUENTS

def get_num_assets():
    return len(get_all_tickers())

def get_num_strikes():
    return len(MONEYNESS_GRID)

# Monte Carlo settings
N_MC_PATHS = 50000
MC_HORIZON_DAYS = 63  # 3 months ≈ 63 trading days
MC_SEED = 42
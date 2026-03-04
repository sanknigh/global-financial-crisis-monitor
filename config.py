# ==============================
# GLOBAL CONFIGURATION
# ==============================

# Data
START_DATE = "2000-01-01"
END_DATE = None  # None = latest available
FREQUENCY = "ME"  # Monthly end

# Windowing
WINDOW_SIZE = 12  # 12 months rolling window

# Train/Test Split
TRAIN_RATIO = 0.7
RANDOM_STATE = 42

# Crisis Periods (Real Historical Events)
CRISIS_PERIODS = [
    ("2008-01-01", "2009-12-31"),  # Global Financial Crisis
    ("2011-01-01", "2012-12-31"),  # Euro Debt Crisis
    ("2020-01-01", "2020-12-31"),  # COVID
    ("2022-01-01", "2022-12-31"),  # Inflation Shock
]

# Model Parameters
LSTM_HIDDEN_SIZE = 64
DENSE_UNITS = 32
DROPOUT = 0.3

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
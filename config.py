# ==========================================
# Statmize Health Intelligence Engine Config
# ==========================================

# =====================================================
# 1️⃣  DATA-DRIVEN PHYSIOLOGICAL CALIBRATION SETTINGS
# =====================================================

# Label used to detect resting baseline
REST_LABEL = "Rest"

# Labels considered high-intensity for peak HR detection
HIGH_INTENSITY_LABELS = ["Running", "Exercise"]

# Percentile used to estimate maximum HR (data-driven)
MAX_HR_PERCENTILE = 95


# =====================================================
# 2️⃣  STRAIN FORMULA WEIGHTS (Hybrid Layer)
# =====================================================

STRAIN_WEIGHTS = {
    "hr_reserve_load": 0.45,     # Cardiovascular load via HR reserve
    "hrv_suppression": 0.30,     # Nervous system fatigue
    "spo2_drop": 0.15,           # Oxygen efficiency drop
    "recovery_delay": 0.10       # Recovery slope penalty
}


# =====================================================
# 3️⃣  STRAIN LEVEL THRESHOLDS (0–100 SCALE)
# =====================================================

STRAIN_THRESHOLDS = {
    "low": 0,
    "moderate": 40,
    "high": 65,
    "critical": 85
}


# =====================================================
# 4️⃣  RISK DETECTION THRESHOLDS
# =====================================================

# HR Reserve over this value indicates overexertion
OVEREXERTION_HR_RESERVE = 0.90

# HRV drop percentage indicating nervous system strain
LOW_HRV_PERCENT_DROP = 0.50

# Critical oxygen saturation
CRITICAL_SPO2 = 92.0

# Warning oxygen threshold
WARNING_SPO2 = 94.0


# =====================================================
# 5️⃣  ROLLING FEATURE WINDOWS
# =====================================================

# Window sizes for smoothing and recovery analysis
ROLLING_WINDOW_SECONDS = 60
RECOVERY_ANALYSIS_WINDOW = 120


# =====================================================
# 6️⃣  MODEL SETTINGS (ML Layer)
# =====================================================

MODEL_TYPE = "lightgbm"  # options: "lightgbm", "random_forest"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Whether to standardize numerical features
USE_FEATURE_SCALING = True


# =====================================================
# 7️⃣  OUTPUT SETTINGS
# =====================================================

STRAIN_SCALE_MIN = 0
STRAIN_SCALE_MAX = 100

ENABLE_HYDRATION_ALERT = True
ENABLE_RECOVERY_STATUS = True


# =====================================================
# 8️⃣  DEBUG / DEVELOPMENT FLAGS
# =====================================================

DEBUG_MODE = True
LOG_INTERMEDIATE_VALUES = False
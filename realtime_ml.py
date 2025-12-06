import joblib
import time
import numpy as np
import json
import pandas as pd
from datetime import datetime


# ============================================================
# 1. LOAD MODEL + LOAD DATASET
# ============================================================

model = joblib.load("isolation_forest_new.pkl")
df = pd.read_csv("test_data.csv")

index = 0  # pointer for streaming CSV rows


# ============================================================
# 2. GLOBAL VARIABLES FOR ALERT ENGINE
# ============================================================

voltage_medium_start = None
voltage_high_start = None

current_medium_start = None
current_high_start = None

last_voltage_alert_time = 0
last_current_alert_time = 0

# Normal ranges (you can adjust)
VOLTAGE_NORMAL_MIN = 215
VOLTAGE_NORMAL_MAX = 245

CURRENT_NORMAL_MIN = 0.6
CURRENT_NORMAL_MAX = 2.0


# ============================================================
# 3. DATA STREAMING FROM CSV
# ============================================================

def get_new_data():
    """Return next row from CSV, cyclically."""
    global index
    row = df.iloc[index]
    index = (index + 1) % len(df)
    return [row.voltage, row.current, row.power, row.energy_Wh]


# ============================================================
# 4. SEVERITY BASED ON SCORE
# ============================================================

def get_severity(score):
    if score > 0:
        return "NORMAL"
    elif score > -0.1:
        return "LOW"
    elif score > -0.3:
        return "MEDIUM"
    elif score > -0.6:
        return "HIGH"
    else:
        return "CRITICAL"


# ============================================================
# 5. DETERMINE WHETHER ANOMALY IS VOLTAGE OR CURRENT
# ============================================================

def get_anomaly_source(voltage, current):
    if voltage < VOLTAGE_NORMAL_MIN or voltage > VOLTAGE_NORMAL_MAX:
        return "voltage"
    if current < CURRENT_NORMAL_MIN or current > CURRENT_NORMAL_MAX:
        return "current"
    return None


# ============================================================
# 6. ALERT ENGINE (MEDIUM 3 MIN / HIGH 10 SEC)
# ============================================================

def check_alerts(voltage, current, severity):
    global voltage_medium_start, voltage_high_start
    global current_medium_start, current_high_start
    global last_voltage_alert_time, last_current_alert_time

    now = time.time()
    source = get_anomaly_source(voltage, current)

    # No anomaly → reset timers
    # If severity becomes NORMAL → reset ALL anomaly timers
    if severity == "NORMAL":
        voltage_medium_start = None
        voltage_high_start = None
        current_medium_start = None
        current_high_start = None
        return None


    # ---------------- VOLTAGE ALERT LOGIC ------------------
    if source == "voltage":

        # MEDIUM severity → wait 3 minutes
        if severity == "MEDIUM":
            if voltage_medium_start is None:
                voltage_medium_start = now
            elif now - voltage_medium_start >= 180:  # 3 minutes
                if now - last_voltage_alert_time > 30:
                    last_voltage_alert_time = now
                    return "⚠ Voltage Drop/Spike (MEDIUM severity >3 min)"


        # HIGH or CRITICAL severity → wait 10 seconds
        if severity in ["HIGH", "CRITICAL"]:
            if voltage_high_start is None:
                voltage_high_start = now
            elif now - voltage_high_start >= 10:
                if now - last_voltage_alert_time > 30:
                    last_voltage_alert_time = now
                    return "🚨 URGENT: Voltage Anomaly (High/Critical)"


    # ---------------- CURRENT ALERT LOGIC ------------------
    if source == "current":

        if severity == "MEDIUM":
            if current_medium_start is None:
                current_medium_start = now
            elif now - current_medium_start >= 180:
                if now - last_current_alert_time > 30:
                    last_current_alert_time = now
                    return "⚠ Current Drop/Spike (MEDIUM severity >3 min)"

        if severity in ["HIGH", "CRITICAL"]:
            if current_high_start is None:
                current_high_start = now
            elif now - current_high_start >= 10:
                if now - last_current_alert_time > 30:
                    last_current_alert_time = now
                    return "🚨 URGENT: Current Anomaly (High/Critical)"

    return None


# ============================================================
# 7. REAL-TIME LOOP
# ============================================================

print("\nReal-time ML Monitoring Started...\nPress CTRL+C to stop.\n")

while True:

    # 1. Get next row
    X = np.array([get_new_data()])
    voltage, current, power, energy = X[0]

    # 2. ML prediction
    y_pred_raw = model.predict(X)
    score = model.decision_function(X)[0]

    anomaly_flag = 1 if y_pred_raw[0] == -1 else 0
    severity = get_severity(score)

    # 3. Check alerts
    alert_message = check_alerts(voltage, current, severity)

    # 4. Form response packet
    output = {
        "timestamp": datetime.now().isoformat(),
        "voltage": float(voltage),
        "current": float(current),
        "power": float(power),
        "energy_Wh": float(energy),
        "anomaly": anomaly_flag,
        "score": float(score),
        "severity": severity
    }

    print(json.dumps(output, indent=4))

    # 5. Print alert if triggered
    if alert_message:
        print("\n==================== ALERT ====================\n")
        print(alert_message)
        print("\n==============================================\n")

    time.sleep(1)  # simulate 1-second IoT sensor feed

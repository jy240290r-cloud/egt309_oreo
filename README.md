# EGT309 Project – Group Oreo

## Group Members
- Cassandra Violette
- Chia Jia Yi
- Yanoria Jhino

## Setup
Install the required packages:
```
pip install -r requirements.txt
```

## EDA
Open `eda.ipynb` in VS Code or Jupyter. Data is read from `data/proj_machine_readings.csv`.

## Summary of Key EDA Findings

The dataset has 100,000 sensor readings from 50 machines, one reading per minute from 1 Jan 2025 to 11 Mar 2025. No duplicate rows were found.

1. The target is imbalanced (80 / 20).
   About 80.3% of readings do not need maintenance and 19.7% do. A model that always predicts "no maintenance" would already be 80% accurate, so accuracy is not a useful metric for this problem (see Choice of Metrics).

2. Temperature has 14% missing values (14,000 rows).
   - That is too many rows to drop, so the values are imputed.
   - The missing rate is the same for both target classes and for all machines, so the values appear to be missing at random.
   - We impute with the median. The mean (75.03) and median (75.09) are almost the same because the distribution is symmetric, but the median is not affected by the extreme overheating values, which we keep.
   - KNN imputation was not used because temperature is not correlated with any other sensor.
   - The imputation is fitted on the training set only, to avoid data leakage.

3. Vibration has 6.7% negative values, which are invalid.
   A vibration level cannot be negative. When the sign is flipped, the negative values have the same distribution and the same maintenance rate as the positive values. Values below -80 also behave exactly like values above +80 (100% need maintenance). We concluded these are sign errors and fixed them with abs().

4. Outliers are kept on purpose.
   The IQR rule flags 649 temperature outliers, but these high readings are exactly the overheating cases the alert system needs to catch. Removing them would remove the signal.

5. Temperature and vibration work like thresholds.
   - The maintenance rate stays at about 14% for temperatures up to 90 °C, then jumps to 100% above 90 °C.
   - Vibration stays at about 18% up to 80 mm/s, then jumps to 100% above 80 mm/s.

6. Humidity, pressure, energy consumption, time and machine ID carry almost no signal.
   - Humidity, pressure and energy consumption are evenly spread across their range, and the maintenance rate barely changes across them.
   - The maintenance rate is flat across hour of day, day of week and machine.
   - Consecutive readings from the same machine are independent, so rolling-window and lag features were not useful.

7. The derived columns leak the target.
   maintenance_required is exactly 1 when machine_status = 2 (failure), anomaly_flag = 1, or predicted_remaining_life < 20. This rule matches all 100,000 rows. failure_type and downtime_risk carry the same information.
   These columns are produced by the same logic as the label, or only exist after the machine has failed, so we exclude them from the model features. A quick model that used them scored a perfect 1.0, which shows the leakage rather than any real predictive ability.

8. Sensors alone can only catch part of the maintenance cases.
   Only about 41–45% of maintenance cases show up in the sensor readings (overheating or high vibration). The rest are machines that had already failed or had low remaining life, and these cannot be seen from the sensors at that moment. A quick sensor-only Random Forest reached about 86% precision but only 42% recall. This is a limit of the data, not of the model.

## Feature Engineering

vibration_abs
- How: absolute value of vibration.
- Why: fixes the invalid negative readings (sign errors), so a -95 reading is correctly treated as high vibration (95).

temperature_missing
- How: 1 if temperature was missing, otherwise 0.
- Why: tells the model which temperature values were imputed and not real, so it can rely on vibration for those rows.

temp_over_90
- How: 1 if temperature is above 90 °C, otherwise 0.
- Why: the maintenance rate jumps from about 14% to 100% above 90 °C. A linear model like Logistic Regression cannot learn a sudden jump on its own, so this flag lets it capture the threshold.

vib_over_80
- How: 1 if vibration_abs is above 80 mm/s, otherwise 0.
- Why: same reason. The maintenance rate jumps from about 18% to 100% above 80 mm/s.

The thresholds (90 and 80) were taken from the bin tables in the EDA, where the maintenance rate jumps to 100%.

Features we tested and dropped:
- Rolling means and lag features: no information, because readings are independent over time.
- Temperature deviation from each machine's median: no extra information, because all machines behave the same.
- Hour and day of week: no information.

These were compared using mutual information with the target (Section 7 of eda.ipynb).

## Choice of Metrics

Because the data is 80 / 20 imbalanced, accuracy is not used. Predicting "no maintenance" for every reading already scores 80.3% while catching zero machines.

Recall (class 1) is our main metric.
It measures how many of the machines that really needed maintenance we caught. A missed machine (false negative) leads to unscheduled downtime, which is the main thing the company wants to prevent.

Precision (class 1)
It measures how many of the alerts we raised were real. Too many false alarms waste the maintenance team's time and make them stop trusting the alerts.

F1 (class 1)
It balances recall and precision. It stops a model from scoring well by flagging every machine, which would give 100% recall but very poor precision.

ROC-AUC / PR-AUC
These measure performance across all decision thresholds, so models can be compared before a final threshold is chosen.

We focus on recall because, for this problem, a missed maintenance case costs more than a false alarm. F1 is reported alongside it so that recall is not achieved by simply flagging everything.
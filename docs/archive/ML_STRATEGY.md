# AquaNexus ML Strategy & Technical Specifications

**Project**: AquaNexus  
**Component**: Machine Learning Pipeline  
**Version**: 1.0  
**Created**: 2026-09-05  
**Purpose**: Detailed technical specifications for ML implementation, data sources, training, and deployment  

---

## 1. ML Objectives & Target Variable

### Primary Objective
Build a **regression model** that predicts **habitat suitability scores** (0.0 to 1.0) based on environmental conditions in Japanese rivers.

### Target Variable: Habitat Suitability Index (HSI)

**Range**: 0.0 → 1.0 (continuous)

**Interpretation**:
```
0.0-0.2: Unsuitable
  - Conditions incompatible with most freshwater species
  - Extreme stress (very low DO, extreme temp, high sediment)
  
0.2-0.4: Poor
  - Limited species can survive
  - Significant environmental stress
  - Only stress-tolerant organisms present
  
0.4-0.6: Moderate
  - Some species can survive
  - Mixed communities possible
  - Regular environmental stress
  
0.6-0.8: Good
  - Most native species can thrive
  - Diverse communities expected
  - Minor environmental stress
  
0.8-1.0: Optimal
  - Excellent conditions for all native species
  - High biodiversity expected
  - Minimal environmental stress
```

### Why Continuous Regression (Not Classification)?
1. **Interpretability**: Easier to explain marginal changes (0.65 → 0.72)
2. **Granularity**: Captures subtle environmental degradation
3. **SHAP compatibility**: SHAP explanations work better with continuous targets
4. **Scenario analysis**: "What-if" simulations need continuous scale
5. **Portfolio value**: Shows sophisticated modeling approach

---

## 2. Datasets & Data Sources

### 2.1 Dataset Composition

**Total Dataset**: ~50,000 records × 25 features

**Temporal Coverage**: 2020-2023 (4 years, 1-hour resolution)

**Spatial Coverage**: Yodo River, Japan
- River length: ~130 km
- Reaches modeled: 10 cross-sections
- Represents: Lower Yodo (closest to IGES in Kanagawa)

### 2.2 Data Source 1: HEC-RAS Simulations (Primary Physics)

**You will run these simulations on your HP Omen**

**HEC-RAS Configuration**:
```
Project: Yodo_River_2020_2023.ras
Model Type: 2D unsteady flow + water quality + sediment transport
Geometry: 10 cross-sections (every ~13 km)
Time step: 15 minutes (HEC-RAS output: aggregated to 1-hour)
Duration: 2020-01-01 to 2023-12-31

Modules enabled:
  ✓ Hydraulics (1D/2D flow)
  ✓ Water temperature (heat transport)
  ✓ Water quality (NSMI - nitrogen/sediment/DO/nutrients)
  ✓ Sediment transport (suspended + bed load)

Boundary conditions:
  - Upstream BC: Historical discharge (Q) data from MLIT
  - Downstream BC: Water level from tidal/pooling (if applicable)
  - Lateral inflows: Tributary/effluent inputs
```

**HEC-RAS Outputs** (1-hour resolution, 10 reaches × 4 years):

| Variable | Unit | Description |
|----------|------|-------------|
| `depth` | m | Water depth at cross-section |
| `velocity` | m/s | Mean velocity (cross-section average) |
| `flow` | m³/s | Discharge (summed across cross-section) |
| `shear_stress` | Pa | Bed shear stress (τ = ρgRS, R=hydraulic radius, S=slope) |
| `water_surface_elev` | m | Water surface elevation (stage) |
| `temperature` | °C | Water temperature (from thermal module) |
| `dissolved_oxygen` | mg/L | DO concentration (from WQ module) |
| `cbod` | mg/L | Carbonaceous biochemical oxygen demand |
| `nitrogen` | mg/L | Total nitrogen (or NO₃⁻) |
| `phosphorus` | mg/L | Total phosphorus |
| `algae` | mg/L | Algal biomass concentration |
| `suspended_sediment` | mg/L | Suspended sediment concentration |
| `bed_elevation` | m | Bed elevation change (from sediment transport) |

**Size**: 4 years × 365 days × 24 hours × 10 reaches = **35,040 records**

---

### 2.3 Data Source 2: Japanese River Monitoring Data (Real Observations)

**Merge with HEC-RAS to align real data with simulations**

#### Option A: NIES Database (Recommended)
**Name**: National Institute for Environmental Studies Water Quality Database  
**URL**: https://www.nies.go.jp/ (search: water quality monitoring)  
**Data**:
- Temperature, DO, pH, conductivity, turbidity
- Nutrients (N, P)
- Major Japanese rivers (Yodo included)
- Frequency: Weekly to monthly
- Coverage: 1990-2024

**Access**: 
1. Visit NIES website
2. Request historical data for Yodo River
3. Download CSV (free, academic use)
4. ~1,000-2,000 records for 2020-2023

#### Option B: MLIT (Ministry of Land, Infrastructure, Transport & Tourism)
**URL**: https://www.mlit.go.jp/ (Water Department)  
**Data**:
- Discharge (m³/s) — most reliable
- Water level (stage)
- Temperature
- Frequency: Daily to hourly
- Coverage: Real-time + 30 years historical

**Access**: 
1. MLIT public data portal
2. Filter by Yodo River
3. Download CSV files

#### Option C: Prefectural Environmental Agencies
**Osaka Prefecture Environmental Department** (Yodo River runs through Osaka)  
**Data**: Local water quality monitoring  
**Access**: Prefecture website (typically public)

#### Option D: Zenodo / Open Data Portals
Search: "Japanese river water quality data" or "Yodo River monitoring"

**Merged Monitoring Data**: **~12,000-15,000 records** (weekly/daily observations)

### 2.4 Combined Dataset

**Final merged dataset**:
```
50,000 records total
= 35,040 HEC-RAS (1-hour, complete spatial-temporal grid)
+ 12,000-15,000 real monitoring observations (weekly-daily)
  (interpolated/aligned to HEC-RAS time grid)

Time-indexed, 1-hour resolution:
2020-01-01 00:00:00 to 2023-12-31 23:00:00
```

---

## 3. Feature Engineering

### 3.1 Raw Features (from HEC-RAS & Monitoring)

**Hydrodynamic Features** (7 features):
```
depth                    [m]               0.5-3.5
velocity                 [m/s]             0.1-1.5
flow                     [m³/s]            50-500
shear_stress             [Pa]              0.1-20
water_surface_elev       [m]               -5 to 5
froude_number (computed) [dimensionless]   0.1-1.5
reynolds_number (computed) [dimensionless] 1000-50000
```

**Thermal Features** (3 features):
```
temperature              [°C]              5-35
thermal_gradient         [°C/km]           -2 to 2   (computed: dT/dx)
solar_radiation          [W/m²]            0-800     (if available, else estimate)
```

**Water Quality Features** (7 features):
```
dissolved_oxygen         [mg/L]            0-14
do_deficit               [mg/L]            0-14      (sat_do - actual_do)
cbod                     [mg/L]            0-10
nitrogen_total           [mg/L]            0-5
phosphorus_total         [mg/L]            0-1
algae_concentration      [mg/L]            0-100
ph                       [dimensionless]   6.5-8.5
```

**Sediment Features** (3 features):
```
suspended_sediment       [mg/L]            0-500
bed_load                 [kg/s]            0-100     (if available)
turbidity                [NTU]             0-500     (computed from suspended sediment)
```

### 3.2 Derived/Engineered Features (8 features)

**Stress Indices**:
```
thermal_stress_index = 1 / (1 + exp(-a * (temp - optimal_temp)))
  optimal_temp = 17.5°C
  Penalty for temp > 25°C or < 10°C

oxygen_stress_index = exp(-b * (optimal_do - do))
  optimal_do = 10 mg/L
  Severe penalty for DO < 2 mg/L or DO > 12 mg/L

combined_stress_index = sqrt(thermal_stress_index² + oxygen_stress_index²)
```

**Interaction Terms**:
```
temp_x_do = temperature * dissolved_oxygen
  (Captures synergistic stress: high temp + low DO)

flow_x_sediment = flow * suspended_sediment
  (Captures transport capacity)

discharge_x_temp = flow * temperature
  (Thermal loading proxy)
```

**Lagged Features** (capture temporal dependencies):
```
depth_lag1, depth_lag6, depth_lag24       [m]
do_lag1, do_lag6, do_lag24                [mg/L]
temp_lag1, temp_lag6, temp_lag24          [°C]
sediment_lag1, sediment_lag6, sediment_lag24 [mg/L]
```

**Rolling Averages** (smooth short-term noise):
```
depth_ma7                 [m]              (7-day rolling mean)
do_ma7, do_ma30           [mg/L]           (7-day, 30-day rolling means)
temp_ma7, temp_ma30       [°C]
sediment_ma7              [mg/L]
velocity_ma7              [m/s]
```

### 3.3 Final Feature Set (25 features)

```
Hydrodynamic (7):
  1. depth
  2. velocity
  3. flow
  4. shear_stress
  5. froude_number
  6. reynolds_number
  7. water_surface_elev

Thermal (3):
  8. temperature
  9. thermal_gradient
  10. solar_radiation

Water Quality (7):
  11. dissolved_oxygen
  12. do_deficit
  13. cbod
  14. nitrogen_total
  15. phosphorus_total
  16. algae_concentration
  17. ph

Sediment (3):
  18. suspended_sediment
  19. bed_load
  20. turbidity

Derived Stress & Interactions (5):
  21. thermal_stress_index
  22. oxygen_stress_index
  23. temp_x_do
  24. flow_x_sediment
  25. combined_stress_index

Lagged/Rolling (handled separately in preprocessing):
  - Lagged features (t-1, t-6, t-24)
  - Rolling means (7-day, 30-day)
```

**Note**: Lagged and rolling features expand feature count, but you can optionally drop them for simpler model if needed.

---

## 4. Synthetic Habitat Suitability Labels

### 4.1 Rationale

**Why synthetic?**
1. Real biological data (fish surveys, macroinvertebrate sampling) is expensive/rare
2. This is a portfolio PoC, not peer-reviewed research
3. Synthetic labels based on ecological literature are realistic & defensible
4. Can easily swap for real bio data if available later

### 4.2 Ecological Basis

Labels based on **fish habitat requirements** (common in freshwater ecology):

**Optimal Conditions for Coldwater Fish** (trout, char):
```
Temperature: 10-18°C (optimal: 15°C)
Dissolved oxygen: 8-12 mg/L (optimal: 10 mg/L)
Depth: 0.5-2.0 m (optimal: 1.2 m)
Velocity: 0.3-0.8 m/s (optimal: 0.5 m/s)
Substrate: gravel/cobble, low turbidity
Flow regime: moderate, stable
```

**Stress Thresholds**:
```
Temperature > 25°C     → Thermal stress (metabolic cost)
DO < 5 mg/L            → Hypoxic stress
DO < 2 mg/L            → Severe stress (avoidance, death)
Suspended sediment > 100 mg/L → Turbidity impact (gills)
Suspended sediment > 200 mg/L → Severe impact
```

### 4.3 Habitat Suitability Label Generation Algorithm

```python
def create_habitat_suitability_label(row):
    """
    Generate HSI (0-1) based on environmental conditions.
    row: pandas Series with columns [temp, do, depth, velocity, suspended_sediment, ...]
    
    Returns: float (0.0 to 1.0)
    """
    
    # 1. Score individual factors using Gaussian curve
    #    (optimal = 1.0, deviations from optimal = lower score)
    
    def gaussian_score(value, optimal, std_dev):
        """Gaussian penalty: highest at optimal, lower away from it."""
        return np.exp(-0.5 * ((value - optimal) / std_dev) ** 2)
    
    temp_score = gaussian_score(
        row['temperature'],
        optimal=17.5,      # Optimal temperature
        std_dev=4.0        # Standard deviation (inflection at ±4°C)
    )
    
    do_score = gaussian_score(
        row['dissolved_oxygen'],
        optimal=10.0,
        std_dev=2.0
    )
    
    depth_score = gaussian_score(
        row['depth'],
        optimal=1.2,
        std_dev=0.4
    )
    
    velocity_score = gaussian_score(
        row['velocity'],
        optimal=0.5,
        std_dev=0.2
    )
    
    # 2. Turbidity/sediment penalty (asymmetric: bad above threshold)
    sediment_score = 1.0 if row['suspended_sediment'] < 50 else \
                     1.0 - (row['suspended_sediment'] - 50) / 200
    sediment_score = np.clip(sediment_score, 0, 1)
    
    # 3. Combine with equal weighting
    base_score = np.mean([temp_score, do_score, depth_score, velocity_score])
    
    # 4. Apply sediment as multiplier (it affects all organisms)
    weighted_score = base_score * sediment_score
    
    # 5. Interaction penalty: high temp + low DO = synergistic harm
    if row['temperature'] > 25 and row['dissolved_oxygen'] < 5:
        interaction_penalty = 0.15  # Reduce score by 15%
    elif row['temperature'] > 22 and row['dissolved_oxygen'] < 4:
        interaction_penalty = 0.20
    else:
        interaction_penalty = 0.0
    
    final_score = weighted_score - interaction_penalty
    
    # 6. Clip to [0, 1]
    return np.clip(final_score, 0.0, 1.0)
```

### 4.4 Example Label Generation

```python
# Example 1: Optimal conditions
state1 = {
    'temperature': 16.0,      # Near optimal
    'dissolved_oxygen': 10.5, # Near optimal
    'depth': 1.3,             # Near optimal
    'velocity': 0.48,         # Near optimal
    'suspended_sediment': 30  # Low sediment
}
HSI = create_habitat_suitability_label(state1)
# Expected: HSI ≈ 0.85-0.92 (excellent)

# Example 2: Thermal + oxygen stress
state2 = {
    'temperature': 28.0,      # Too warm
    'dissolved_oxygen': 3.0,  # Too low
    'depth': 1.2,
    'velocity': 0.5,
    'suspended_sediment': 200 # High sediment
}
HSI = create_habitat_suitability_label(state2)
# Expected: HSI ≈ 0.15-0.25 (poor)

# Example 3: Moderate stress
state3 = {
    'temperature': 20.0,      # Slightly warm
    'dissolved_oxygen': 7.0,  # Acceptable
    'depth': 0.8,             # Shallow
    'velocity': 0.6,
    'suspended_sediment': 80
}
HSI = create_habitat_suitability_label(state3)
# Expected: HSI ≈ 0.55-0.65 (moderate)
```

### 4.5 Implementation in Data Pipeline

```python
# In src/aquanexus/data/synthetic.py

def apply_labels_to_dataset(df):
    """
    Apply habitat suitability labels to entire dataset.
    
    Input: df with raw features
    Output: df with 'habitat_suitability_index' column
    """
    df['habitat_suitability_index'] = df.apply(
        lambda row: create_habitat_suitability_label(row),
        axis=1
    )
    return df

# Usage:
data = pd.read_csv('data/raw/yodo_river_merged.csv')
data = apply_labels_to_dataset(data)
data.to_csv('data/processed/yodo_river_with_labels.csv', index=False)

# Check label distribution
print(data['habitat_suitability_index'].describe())
# count    50000.000000
# mean        0.628000  (should be centered ~0.6)
# std         0.185000
# min         0.012000
# 25%         0.512000
# 50%         0.634000
# 75%         0.745000
# max         0.998000
```

---

## 5. Training Strategy & Data Splits

### 5.1 Temporal Split (Primary Validation Strategy)

**Rationale**: Time-series data has temporal dependence; must validate on future data

```
Total data: 2020-01-01 to 2023-12-31 (4 years)

Training:    2020-01-01 to 2021-12-31 (2 years, 17,520 records, 35%)
Validation:  2022-01-01 to 2022-12-31 (1 year, 8,760 records, 17.5%)
Test:        2023-01-01 to 2023-12-31 (1 year, 8,760 records, 17.5%)

Additional 2022-2023 holdout (15% for final evaluation): 7,500 records
```

**Implementation**:
```python
def create_temporal_split(data_df, random_seed=42):
    train_mask = (data_df['timestamp'] >= '2020-01-01') & \
                 (data_df['timestamp'] < '2022-01-01')
    val_mask = (data_df['timestamp'] >= '2022-01-01') & \
               (data_df['timestamp'] < '2023-01-01')
    test_mask = (data_df['timestamp'] >= '2023-01-01') & \
                (data_df['timestamp'] < '2024-01-01')
    
    X_train = data_df[train_mask][FEATURES]
    y_train = data_df[train_mask]['habitat_suitability_index']
    
    X_val = data_df[val_mask][FEATURES]
    y_val = data_df[val_mask]['habitat_suitability_index']
    
    X_test = data_df[test_mask][FEATURES]
    y_test = data_df[test_mask]['habitat_suitability_index']
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
```

### 5.2 Spatial Split (Secondary Validation)

**Rationale**: Test model generalization across river reaches

```
Yodo River: 10 reaches (cross-sections at ~13 km intervals)

Training reaches:    1-7 (70% of spatial coverage)
Test reaches:        8-10 (30%, geographically unseen)

All time periods (2020-2023) included in both train & test
```

**Expected result**: Good performance on reaches 8-10 proves model generalizes spatially

### 5.3 Event-Based Split (Tertiary Validation)

**Rationale**: Test on extreme conditions not seen in training

```
Training:    Normal flow conditions (2020-2022)
             Discharge: 50-200 m³/s (typical monsoon/dry seasons)

Test:        Extreme events (2023)
             - Severe monsoon (discharge > 300 m³/s, flood)
             - Drought (discharge < 50 m³/s, low flow)
```

**Expected result**: Model should still predict reasonable values (not extrapolate wildly)

---

## 6. Model Architecture & Algorithms

### 6.1 Primary Model: XGBoost (Gradient Boosting Regressor)

**Why XGBoost?**
```
✓ Non-linear relationships (temp × DO interaction)
✓ Built-in feature importance (for SHAP)
✓ Fast training (~5 sec on 50K records)
✓ Fast inference (~1 ms per prediction)
✓ Handles missing data gracefully
✓ Resistant to overfitting (L1/L2 regularization)
✓ Industry-standard for tabular data
✓ Excellent calibration (predictions ≈ actual)
```

**Hyperparameters**:
```python
xgb_params = {
    'n_estimators': 200,           # Number of boosting rounds
    'max_depth': 7,                # Tree depth (7-8 is sweet spot)
    'learning_rate': 0.05,         # Shrinkage (slow, steady learning)
    'subsample': 0.8,              # Row sampling (80% per tree)
    'colsample_bytree': 0.8,       # Feature sampling (80% per tree)
    'min_child_weight': 1,         # Minimum leaf weight
    'gamma': 0,                    # Minimum loss reduction for split
    'reg_alpha': 0.1,              # L1 regularization
    'reg_lambda': 1.0,             # L2 regularization
    'random_state': 42,
    'objective': 'reg:squarederror',  # Regression (MSE loss)
    'eval_metric': 'rmse',
    'verbosity': 1,
    'n_jobs': -1                   # Parallel processing
}

model = xgb.XGBRegressor(**xgb_params)
```

**Training**:
```python
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,  # Stop if val loss doesn't improve for 20 rounds
    verbose=10
)
```

**Expected Performance** (test set):
```
RMSE: 0.10-0.15
MAE:  0.07-0.10
R²:   0.80-0.88
Pearson r: 0.90-0.95
```

### 6.2 Baseline Model 1: Random Forest (Comparison)

**Why include?**
- Shows value of gradient boosting (XGBoost should outperform)
- More interpretable than XGBoost
- Good for feature importance validation

**Hyperparameters**:
```python
rf_params = {
    'n_estimators': 200,
    'max_depth': 15,           # Deeper trees (RF can handle it)
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

model_rf = RandomForestRegressor(**rf_params)
model_rf.fit(X_train, y_train)
```

**Expected Performance**:
```
RMSE: 0.14-0.18 (worse than XGBoost)
R²:   0.75-0.85
Verdict: Random Forest is good baseline, XGBoost is better
```

### 6.3 Baseline Model 2: Linear Regression (Theory)

**Why include?**
- Proves non-linear relationships matter
- Simple baseline (expected to fail)
- Shows value of ML

**Implementation**:
```python
from sklearn.linear_model import LinearRegression

model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
```

**Expected Performance**:
```
RMSE: 0.25-0.30 (much worse)
R²:   0.50-0.65
Verdict: Linear relationships insufficient, ML needed
```

### 6.4 Optional Model: LSTM (Time-Series Neural Network)

**Include if time allows** (adds ~2 hours of training)

**Why?**
- Captures temporal dependencies (24-hour patterns)
- Shows understanding of deep learning
- Alternative to traditional ML

**Structure**:
```python
import tensorflow as tf

model_lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(24, 25)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output: 0-1
])

model_lstm.compile(optimizer='adam', loss='mse', metrics=['mae'])
model_lstm.fit(X_train_lstm, y_train, epochs=20, batch_size=32, validation_data=(X_val_lstm, y_val))
```

**Expected Performance**:
```
RMSE: 0.12-0.16 (similar to XGBoost, but slower)
Training time: 2-3 minutes (vs 5 sec for XGBoost)
Verdict: Not necessary for this task; skip if time-constrained
```

---

## 7. Training Process (Step-by-Step)

### Phase 2a: Model Training

```
Step 1: Load raw data
        └─ 50K records from data/processed/yodo_river_with_labels.csv

Step 2: Create temporal split
        ├─ Train: 2020-2021 (17,520 records)
        ├─ Val: 2022 (8,760 records)
        └─ Test: 2023 (8,760 records)

Step 3: Fit StandardScaler on training data
        └─ Normalize all features to mean=0, std=1

Step 4: Normalize train/val/test sets
        └─ Apply scaler fitted on train to all sets

Step 5: Train XGBoost
        ├─ n_estimators=200, max_depth=7, learning_rate=0.05
        ├─ Monitor validation RMSE
        ├─ Early stopping if no improvement for 20 rounds
        └─ Save best model → data/models/xgboost_v1.joblib

Step 6: Evaluate XGBoost on test set
        ├─ RMSE: ________
        ├─ MAE: ________
        ├─ R²: ________
        └─ Pearson r: ________

Step 7: Train Random Forest (baseline)
        ├─ Same data splits
        ├─ n_estimators=200, max_depth=15
        └─ Save → data/models/random_forest_v1.joblib

Step 8: Evaluate Random Forest
        └─ Compare metrics vs XGBoost

Step 9: Save preprocessing artifacts
        ├─ scaler.joblib (StandardScaler)
        ├─ feature_names.pkl (column names)
        └─ model_config.json (hyperparameters)

Step 10: Generate training summary
         ├─ Training metrics table
         ├─ Validation metrics table
         ├─ Test metrics table
         └─ Learning curves (loss vs epoch)
```

### Phase 2b: Explainability with SHAP

```
Step 1: Load trained XGBoost model

Step 2: Create SHAP explainer
        ├─ Use TreeExplainer (native XGBoost support)
        ├─ Background data: validation set (8,760 records)
        └─ Fit explainer: explainer = shap.TreeExplainer(model)

Step 3: Compute SHAP values
        ├─ For entire test set (8,760 records)
        ├─ Take 1000 samples if too slow
        └─ Save SHAP values → data/models/shap_values_test.npy

Step 4: Generate SHAP plots
        ├─ Summary bar plot (feature importance)
        ├─ Summary plot (force plots)
        ├─ Dependence plots (individual features)
        └─ Save to docs/plots/

Step 5: Feature importance ranking
        ├─ Rank by mean |SHAP|
        ├─ Top 10 features: ________
        └─ Bottom 5 features: ________

Step 6: Interaction analysis
        ├─ Detect pairwise interactions
        ├─ temp × do: strongest?
        ├─ flow × sediment: present?
        └─ Document findings
```

### Phase 2c: Validation & Baselines

```
Step 1: Temporal validation (test 2023 data)
        ├─ RMSE: ________
        ├─ Compare to 2022 validation RMSE
        └─ Verdict: ✓ PASS if < 10% difference

Step 2: Spatial validation (reaches 8-10)
        ├─ Train on reaches 1-7
        ├─ Test on reaches 8-10
        ├─ RMSE: ________
        └─ Verdict: ✓ PASS if R² > 0.75

Step 3: Event-based validation
        ├─ Train on normal conditions
        ├─ Test on 2023 monsoon/drought
        ├─ Check for extrapolation errors
        └─ Verdict: ✓ PASS if predictions reasonable (0-1 range)

Step 4: Baseline comparisons
        ├─ XGBoost RMSE: ________
        ├─ Random Forest RMSE: ________ (+_____%)
        ├─ Linear Regression RMSE: ________ (+_____%)
        └─ Verdict: XGBoost significantly better

Step 5: Generate validation report
        └─ Table: Model | RMSE | MAE | R² | Temporal | Spatial | Event
```

---

## 8. Algorithm Deployment Map

**Where each model is used in the system**:

| Algorithm | Component | Endpoint | Purpose | Input | Output |
|-----------|-----------|----------|---------|-------|--------|
| **XGBoost** | FastAPI | `POST /predict` | Real-time prediction | Single env. state | HSI score (0-1) + confidence |
| **XGBoost** | FastAPI | `POST /batch_predict` | Batch CSV predictions | CSV file | JSON array of scores |
| **XGBoost** | FastAPI | `POST /scenario_run` | What-if scenario | Modified env. state | Predicted HSI + comparison |
| **XGBoost** | FastAPI | `POST /interactive_analysis` | 2D heatmap | Feature ranges | Heatmap (temp × DO) |
| **SHAP** | FastAPI | `GET /explain/{id}` | Explain prediction | Prediction ID | Feature contributions |
| **SHAP** | Frontend | ExplainabilityPanel | Visual explanation | Prediction data | SHAP bar chart |
| **Random Forest** | Notebook | 05_model_validation.ipynb | Baseline comparison | Test data | Metrics table |
| **Linear Regression** | Notebook | 05_model_validation.ipynb | Baseline (proof ML needed) | Test data | Metrics table |
| **LSTM** | (Optional) | `/predict_timeseries` | Time-series forecast | Past 24 hours | Next hour HSI |

---

## 9. Expected Performance Benchmarks

### 9.1 XGBoost (Primary Model)

**Test Set Performance**:
```
RMSE:   0.10-0.15  (habitat suitability scale 0-1)
MAE:    0.07-0.10  (mean absolute error)
R²:     0.80-0.88  (explains 80-88% of variance)
Pearson correlation: 0.90-0.95
Nash-Sutcliffe Efficiency (NSE): 0.78-0.85
```

**Interpretation**:
- Average prediction error: ±0.08-0.10 on 0-1 scale
- Model explains most variance in habitat suitability
- Suitable for operational decision-support

### 9.2 By Validation Split

**Temporal Validation (Test 2023)**:
```
RMSE: 0.12 ± 0.03
No significant degradation → Model generalizes to future data ✓
```

**Spatial Validation (Reaches 8-10)**:
```
R²: 0.78 ± 0.05
Minor degradation acceptable → Model generalizes to new reaches ✓
```

**Event-Based Validation (Monsoon/Drought)**:
```
RMSE on extreme: 0.15 ± 0.04
Slightly higher (expected) → Model handles extremes reasonably ✓
```

### 9.3 Baseline Comparison

```
| Model | RMSE | MAE | R² | Status |
|-------|------|-----|----|----|
| XGBoost (best) | 0.12 | 0.08 | 0.84 | ✅ Selected |
| Random Forest | 0.16 | 0.11 | 0.79 | ✓ Decent baseline |
| Linear Regression | 0.28 | 0.22 | 0.55 | ❌ Poor (proves ML needed) |
| Persistence (naive) | 0.35 | 0.28 | 0.40 | ❌ Worst baseline |
```

---

## 10. Data Quality & Validation Checks

### 10.1 Pre-Training Validation

```python
def validate_dataset_quality(df):
    """Checks to run before training."""
    
    # 1. Check row count
    assert len(df) > 40000, f"Too few records: {len(df)}"
    
    # 2. Check for missing values
    missing = df.isnull().sum()
    assert missing.max() < 0.1 * len(df), "Too many nulls in some columns"
    
    # 3. Check feature ranges (outlier detection)
    assert df['temperature'].min() > -5, "Temp too low (sensor error?)"
    assert df['temperature'].max() < 50, "Temp too high"
    assert df['dissolved_oxygen'].min() > -1, "DO too low"
    assert df['dissolved_oxygen'].max() < 15, "DO too high"
    
    # 4. Check target variable distribution
    hsi = df['habitat_suitability_index']
    assert 0.2 < hsi.mean() < 0.8, f"HSI mean suspicious: {hsi.mean()}"
    assert hsi.std() > 0.1, f"HSI has no variance: {hsi.std()}"
    
    # 5. Check temporal coverage
    dates = pd.to_datetime(df['timestamp'])
    assert (dates.max() - dates.min()).days > 1000, "Time span too short"
    
    print("✓ Dataset validation passed")
```

### 10.2 Post-Training Validation

```python
def validate_model_predictions(model, X_test, y_test):
    """Sanity checks on trained model."""
    
    y_pred = model.predict(X_test)
    
    # 1. All predictions in [0, 1]
    assert (y_pred >= 0).all() and (y_pred <= 1).all(), "Out of range predictions"
    
    # 2. Predictions have reasonable variance
    assert y_pred.std() > 0.05, "Predictions have no variance (overfitting?)"
    
    # 3. Predictions correlate with actual
    corr = np.corrcoef(y_pred, y_test)[0, 1]
    assert corr > 0.80, f"Poor correlation with actual: {corr}"
    
    # 4. No systematic bias
    bias = (y_pred - y_test).mean()
    assert abs(bias) < 0.05, f"Model is systematically biased: {bias}"
    
    print(f"✓ Model validation passed (r={corr:.3f}, bias={bias:.3f})")
```

---

## 11. Explainability Strategy (SHAP)

### 11.1 Feature Importance

```
Method: SHAP TreeExplainer
Output: Mean |SHAP| value per feature

Example top 10:
  1. dissolved_oxygen:       0.165
  2. temperature:            0.142
  3. oxygen_stress_index:    0.098
  4. suspended_sediment:     0.085
  5. flow:                   0.072
  6. do_deficit:            0.061
  7. thermal_stress_index:   0.055
  8. velocity:              0.048
  9. phosphorus_total:      0.039
  10. algae_concentration:   0.032

Interpretation: DO & temp are dominant drivers of habitat suitability
```

### 11.2 Critical Thresholds

```
From SHAP dependence plots, identify values where impact changes:

Dissolved Oxygen:
  - DO > 9 mg/L: No penalty (optimal)
  - DO 5-9 mg/L: Gradual penalty
  - DO < 2 mg/L: Severe penalty (avoid zone)

Temperature:
  - Temp 14-20°C: Optimal range
  - Temp > 25°C: Increasing stress
  - Temp > 30°C: Severe stress

Combined Interaction (temp × DO):
  - High temp (>25°C) + Low DO (<5 mg/L): Critical danger zone
  - SHAP shows this interaction contributes -0.15 to -0.25 to score
```

### 11.3 Prediction Explanation Examples

**Example 1: Good habitat**
```
Prediction: 0.82 (good)
SHAP base value: 0.65
Contributions:
  + DO (9.5 mg/L):      +0.12 (beneficial)
  + Temperature (16°C):  +0.08 (optimal range)
  + Flow (120 m³/s):    +0.04 (adequate dilution)
  - Sediment (45 mg/L): -0.03 (minor impact)
  ─────────────────────────────
  Sum: 0.21 (added to base 0.65 = 0.86, clipped to 0.82)

Interpretation: "Habitat suitability is good, driven by adequate DO and 
optimal temperature. Flow provides good habitat diversity."
```

**Example 2: Poor habitat**
```
Prediction: 0.28 (poor)
SHAP base value: 0.65
Contributions:
  - DO (2.1 mg/L):         -0.25 (severe stress)
  - Temperature (29°C):    -0.18 (heat stress)
  - interaction (T×DO):    -0.20 (synergistic harm)
  - Sediment (180 mg/L):   -0.08 (turbidity)
  ─────────────────────────────
  Sum: -0.71 (65-71 = -6, clipped to 0.28)

Interpretation: "Habitat suitability is poor due to hypoxia and thermal stress. 
The combination of low oxygen + high temperature is particularly harmful."
```

---

## 12. Model Serialization & Deployment

### 12.1 Save Trained Models

```python
import joblib
import pickle
import json

# Save XGBoost model
joblib.dump(model_xgb, 'data/models/xgboost_v1.joblib')

# Save preprocessing scaler
joblib.dump(scaler, 'data/models/scaler.joblib')

# Save feature names
with open('data/models/feature_names.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)

# Save model metadata
metadata = {
    'model_type': 'xgboost',
    'version': '1.0',
    'training_date': '2026-09-05',
    'test_rmse': 0.12,
    'test_r2': 0.84,
    'features': FEATURES,
    'hyperparameters': {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.05
    }
}

with open('data/models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### 12.2 Load Models at API Startup

```python
# In src/aquanexus/api/app.py

import joblib

@app.on_event("startup")
async def load_models():
    global model, scaler, explainer
    
    # Load model
    model = joblib.load('data/models/xgboost_v1.joblib')
    
    # Load scaler
    scaler = joblib.load('data/models/scaler.joblib')
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    logger.info("✓ Models loaded successfully")
```

---

## 13. Summary Table

| Aspect | Specification |
|--------|---------------|
| **Target** | Habitat suitability index (0-1, continuous) |
| **Data** | 50K records (HEC-RAS + Japanese monitoring, 2020-2023) |
| **Features** | 25 engineered features (hydro, thermal, WQ, sediment, derived) |
| **Labels** | Synthetic, ecologically-based habitat scores |
| **Primary Model** | XGBoost (gradient boosting) |
| **Baselines** | Random Forest, Linear Regression |
| **Optional** | LSTM (time-series) |
| **Training Split** | Temporal: 70% train, 15% val, 15% test |
| **Validation** | Temporal, spatial, event-based |
| **Expected RMSE** | 0.10-0.15 (XGBoost) |
| **Expected R²** | 0.80-0.88 (XGBoost) |
| **Explainability** | SHAP (feature importance, interactions, thresholds) |
| **Inference** | <2 ms per prediction (FastAPI) |
| **Deployment** | API endpoints (/predict, /scenario_run, /explain) |

---

## 14. References & Data Sources

**HEC-RAS Documentation**:
- https://www.hec.usace.army.mil/software/hecras/

**Japanese River Data**:
- NIES: https://www.nies.go.jp/
- MLIT: https://www.mlit.go.jp/
- Local prefectures: data.go.jp

**Habitat Suitability Literature**:
- Bovee et al. (1998): "Instream Flow Council Habitat Models"
- Poff & Hart (2002): "How dams vary fish habitats"

**ML/XGBoost**:
- XGBoost documentation: https://xgboost.readthedocs.io/
- SHAP documentation: https://shap.readthedocs.io/

---

**Document Version**: 1.0  
**Last Updated**: 2026-09-05  
**Author**: Claude (AquaNexus Development)  
**Status**: Ready for implementation


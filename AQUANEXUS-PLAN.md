# AQUANEXUS Development Plan

**Project:** AquaNexus — Physics-Informed ML Framework for Aquatic Ecosystem Diagnosis  
**Timeline:** 4 weeks  
**Scope:** Full-stack proof-of-concept with real HEC-RAS integration, open Japanese river data, trained ML model, FastAPI backend, React frontend  
**Deliverables:** GitHub repo (production-ready code), Streamlit demo, FastAPI + React app, DEVLOG.md tracking, Docker containers  

---

## Architecture Overview

```
Real Japanese River Data
    ↓
HEC-RAS Simulations (.ras files + outputs)
    ↓
Data Pipeline (Python: extraction, preprocessing, validation)
    ↓
Environmental State Vectors
    ↓
Machine Learning Pipeline
    ├─ Model Training (XGBoost/Random Forest/LSTM)
    ├─ Validation (temporal, spatial, event-based splits)
    ├─ Explainability (SHAP, feature importance, interaction analysis)
    └─ Threshold Discovery (critical condition detection)
    ↓
Backend API (FastAPI)
    ├─ Prediction endpoint
    ├─ Scenario simulation endpoint
    ├─ Explainability endpoint
    └─ Batch analysis endpoint
    ↓
Frontend (React)
    ├─ Interactive dashboard
    ├─ Real-time predictions
    ├─ Scenario builder
    ├─ Explainability visualizations
    └─ Diagnosis report generator
```

---

## Phase 1: Foundation & HEC-RAS Integration (Days 1-5)

**Objective:** Build project skeleton, integrate HEC-RAS workflow, establish data pipeline for Japanese river  
**Output:** Working repo with HEC-RAS I/O, data loading, basic structure  
**Success Criteria:** Can read .ras files, simulate river conditions, extract environmental state vectors  

### Phase 1a: Project Setup & Architecture (Day 1)

**Prompt for Claude:**

```
You are building AquaNexus, a proof-of-concept system that integrates HEC-RAS hydraulic 
simulations with machine learning to diagnose aquatic ecosystem stress.

PROJECT SETUP & DIRECTORY STRUCTURE

Create a professional Python project with the following structure:

aquanexus/
├── README.md                          # Project overview for GitHub
├── DEVLOG.md                          # Development log (template provided below)
├── pyproject.toml                     # Python project config (poetry/pip)
├── requirements.txt                   # Dependencies
├── .gitignore                         # Git ignore rules
├── .dockerignore                      # Docker ignore rules
├── Dockerfile                         # Production Docker image
├── docker-compose.yml                 # Local dev Docker setup
│
├── src/
│   └── aquanexus/
│       ├── __init__.py
│       ├── config.py                  # Configuration management (paths, models, parameters)
│       ├── logger.py                  # Logging setup
│       │
│       ├── hecras/
│       │   ├── __init__.py
│       │   ├── reader.py              # Read HEC-RAS .ras files and outputs
│       │   ├── parser.py              # Parse HEC-RAS simulation results
│       │   ├── runner.py              # Execute HEC-RAS simulations programmatically
│       │   └── validator.py           # Validate HEC-RAS input/output
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py              # Load Japanese river monitoring data
│       │   ├── preprocessor.py        # Preprocessing, normalization, feature engineering
│       │   ├── validator.py           # Data quality checks
│       │   └── synthetic.py           # Generate synthetic scenarios
│       │
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── models.py              # Model definitions (XGBoost, RF, LSTM)
│       │   ├── trainer.py             # Training loop, validation, hyperparameter tuning
│       │   ├── explainer.py           # SHAP, feature importance, interaction analysis
│       │   ├── evaluator.py           # Metrics, validation splits, benchmarking
│       │   └── predictions.py         # Inference pipeline, uncertainty quantification
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py                 # FastAPI main app
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── predictions.py     # /predict, /batch_predict endpoints
│       │   │   ├── scenarios.py       # /scenario_run endpoint
│       │   │   ├── explanations.py    # /explain endpoint
│       │   │   └── health.py          # /health endpoint
│       │   ├── schemas.py             # Pydantic models for request/response
│       │   └── middleware.py          # Error handling, logging middleware
│       │
│       ├── dashboard/
│       │   ├── __init__.py
│       │   ├── app.py                 # Streamlit app (quick demo)
│       │   └── utils.py               # Streamlit utilities
│       │
│       └── utils/
│           ├── __init__.py
│           ├── io.py                  # File I/O utilities
│           ├── metrics.py             # Calculation utilities
│           └── viz.py                 # Visualization helpers
│
├── frontend/
│   ├── package.json                   # React dependencies
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── PredictionForm.jsx
│   │   │   ├── ScenarioBuilder.jsx
│   │   │   ├── ExplainabilityPanel.jsx
│   │   │   └── DiagnosisReport.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Predict.jsx
│   │   │   ├── Scenarios.jsx
│   │   │   └── Analysis.jsx
│   │   ├── services/
│   │   │   └── api.js                 # API client (calls FastAPI backend)
│   │   ├── App.jsx
│   │   └── index.js
│   └── .env.example
│
├── notebooks/
│   ├── 01_hecras_workflow.ipynb       # HEC-RAS integration demo
│   ├── 02_data_exploration.ipynb      # Japanese river data analysis
│   ├── 03_model_training.ipynb        # ML model training & validation
│   └── 04_explainability.ipynb        # SHAP & feature analysis
│
├── data/
│   ├── raw/                           # Downloaded Japanese river data
│   ├── processed/                     # Preprocessed data
│   ├── hecras/                        # HEC-RAS .ras files & outputs
│   ├── models/                        # Trained models (pkl, h5, etc.)
│   └── cache/                         # Preprocessed cache for fast loading
│
├── tests/
│   ├── __init__.py
│   ├── test_hecras.py                 # HEC-RAS reader/runner tests
│   ├── test_data.py                   # Data pipeline tests
│   ├── test_ml.py                     # Model tests
│   └── test_api.py                    # API endpoint tests
│
├── docs/
│   ├── ARCHITECTURE.md                # System architecture & data flow
│   ├── HECRAS_GUIDE.md                # How to set up & run HEC-RAS
│   ├── DATA_SOURCES.md                # Japanese river data sources & access
│   ├── ML_METHODOLOGY.md              # ML approach, validation strategy
│   └── API_REFERENCE.md               # FastAPI endpoint documentation
│
└── scripts/
    ├── download_data.py               # Script to download Japanese river data
    ├── run_hecras.py                  # Script to execute HEC-RAS simulations
    ├── preprocess.py                  # Data preprocessing pipeline
    ├── train_models.py                # Model training (can be called standalone)
    └── serve.py                       # Start FastAPI server


TECHNOLOGY STACK

Python:
  - Core: Python 3.11+
  - HEC-RAS integration: subprocess (call HEC-RAS .exe), HECRASController (if available)
  - Data: pandas, numpy, xarray, netCDF4
  - ML: scikit-learn, xgboost, lightgbm, tensorflow/pytorch (optional for LSTM)
  - Explainability: shap, lime, matplotlib, seaborn, plotly
  - Testing: pytest, pytest-cov
  - Logging: python-logging, structlog
  - Config: pydantic, python-dotenv

Backend:
  - API: FastAPI, uvicorn
  - Validation: pydantic
  - Async: asyncio, httpx
  - Database (optional): SQLAlchemy + PostgreSQL for caching predictions

Frontend:
  - React 18
  - TypeScript (for type safety)
  - UI: Material-UI or shadcn/ui
  - State: Redux or Zustand
  - Visualization: Plotly.js, D3.js
  - HTTP: axios

DevOps:
  - Docker & docker-compose
  - GitHub Actions (CI/CD)
  - Uvicorn (production server)


CONFIGURATION & ENVIRONMENT

Create src/aquanexus/config.py:

```python
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "AquaNexus"
    VERSION: str = "0.1.0"
    
    # Paths
    ROOT_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = ROOT_DIR / "data"
    MODELS_DIR: Path = DATA_DIR / "models"
    HECRAS_DIR: Path = DATA_DIR / "hecras"
    
    # HEC-RAS
    HECRAS_EXE: str = "C:\\Program Files\\HEC\\HEC-RAS\\HEC-RASController.exe"  # Windows path
    HECRAS_TIMEOUT: int = 300  # seconds
    
    # ML
    MODEL_TYPE: str = "xgboost"  # xgboost, random_forest, lstm
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.1
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = False
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

Create .env.example:

```
HECRAS_EXE=C:\\Program Files\\HEC\\HEC-RAS\\HEC-RASController.exe
MODEL_TYPE=xgboost
API_DEBUG=false
LOG_LEVEL=INFO
```


INITIALIZE GIT & GITHUB

```bash
cd aquanexus
git init
git add .
git commit -m "Initial project structure and setup

- Project directory structure with src/, frontend/, tests/, docs/
- Configuration management with pydantic settings
- Docker setup for reproducibility
- Requirements placeholder for dependencies

Co-Authored-By: Claude <noreply@anthropic.com>"

# Create GitHub repo (assuming you'll do this):
# gh repo create aquanexus --public --source=. --remote=origin --push
```

OUTPUT CHECKLIST:

✓ Complete directory structure created
✓ pyproject.toml written with metadata
✓ requirements.txt placeholder (we'll add actual deps in Phase 1b)
✓ config.py with environment management
✓ .env.example template
✓ Dockerfile & docker-compose.yml for reproducibility
✓ Initial .gitignore
✓ README.md (project overview)
✓ ARCHITECTURE.md (high-level system design)
✓ Initial git commit

NEXT STEP: Phase 1b (HEC-RAS Integration) will add HEC-RAS reader/runner code.
```

End the prompt with: "Execute this setup completely. Show me the final directory tree and all key files created."

---

### Phase 1b: HEC-RAS Integration & Data Loading (Days 2-3)

**Prompt for Claude:**

```
You are continuing AquaNexus development. Phase 1a foundation is complete.

PHASE 1B: HEC-RAS INTEGRATION & JAPANESE RIVER DATA PIPELINE

OBJECTIVE:
1. Build HEC-RAS reader/runner to read .ras files and extract simulations
2. Implement Japanese river data loader (open-source datasets)
3. Create data preprocessor to merge HEC-RAS outputs with monitoring data
4. Build validation pipeline for data quality

PART 1: HEC-RAS INTEGRATION

Create src/aquanexus/hecras/reader.py to read HEC-RAS .ras project files:

The .ras file is an ASCII text file with sections like:
  [Project]
  [Plan]
  [Geometry]
  [Unsteady Flow Data]
  [Output Options]

Key data to extract:
  - River reach geometry (cross-sections, distances)
  - Flow boundary conditions (upstream/downstream)
  - Manning's coefficients
  - Simulation time steps

Create reader that:
  1. Parses .ras file structure
  2. Extracts flow, geometry, boundary conditions
  3. Returns structured dict with simulation metadata
  4. Validates file integrity

Example function signatures:
```python
def read_ras_file(ras_path: Path) -> Dict[str, Any]:
    '''Parse HEC-RAS .ras project file.'''
    
def extract_simulation_metadata(ras_dict: Dict) -> Dict[str, Any]:
    '''Extract simulation parameters: flow, time steps, geometry.'''
    
def read_hecras_output(output_dir: Path) -> pd.DataFrame:
    '''Read HEC-RAS output files (csv/txt).
    Returns DataFrame with columns: timestamp, reach, depth, velocity, 
                                     temp, do, sediment, ... '''
```

Create src/aquanexus/hecras/runner.py to execute HEC-RAS simulations:

HEC-RAS can be run programmatically via:
  - Command line: HEC-RAS.exe projectname.ras
  - HECRASController (if available on your system)
  - Subprocess calls

Implement:
```python
def run_hecras_simulation(ras_file: Path, timeout: int = 300) -> bool:
    '''Execute HEC-RAS simulation. Returns True if successful.'''
    
def extract_simulation_results(ras_project_dir: Path) -> pd.DataFrame:
    '''After simulation completes, extract results from output files.'''
    
def scenario_run(ras_file: Path, modifications: Dict[str, Any]) -> pd.DataFrame:
    '''Run HEC-RAS with modified parameters (discharge, temperature, etc).
    Useful for "what-if" scenarios.'''
```

PART 2: JAPANESE RIVER DATA LOADING

Recommended open data sources:
  1. NIES (National Institute for Environmental Studies)
     - Water quality monitoring data
     - URL: https://www.nies.go.jp/
  2. Japanese government water data portals
     - MLIT (Ministry of Land, Infrastructure, Transport and Tourism)
     - Prefectural environmental agencies
  3. University databases (e.g., Hiroshima University river monitoring)
  4. Open data portals: data.go.jp, Zenodo

We'll use a combination of:
  - Real river discharge/water level data (USGS-like databases or Japanese equivalents)
  - Water quality data (temperature, DO, suspended sediment, nutrients)
  - Biological observations (fish species presence, macroinvertebrates—if available)

Create src/aquanexus/data/loader.py:

```python
def load_japanese_river_data(
    river_name: str,
    start_date: str,
    end_date: str,
    data_source: str = "nies"
) -> pd.DataFrame:
    '''Load real Japanese river monitoring data.
    
    Returns DataFrame with columns:
      - timestamp
      - discharge (m3/s)
      - water_level (m)
      - temperature (°C)
      - dissolved_oxygen (mg/L)
      - suspended_sediment (mg/L)
      - turbidity (NTU)
      - pH
      - conductivity (μS/cm)
      - nutrients (nitrogen, phosphorus)
      - [optional] biological data (species counts, habitat scores)
    '''

def create_environmental_state_vector(
    hecras_output: pd.DataFrame,
    monitoring_data: pd.DataFrame,
    temporal_align: bool = True
) -> pd.DataFrame:
    '''Merge HEC-RAS simulations with real monitoring data.
    
    Create feature vectors with columns:
      - hydrodynamic: depth, velocity, flow, shear_stress
      - thermal: temperature, thermal_gradient
      - water_quality: do, cbod, nutrients, algae
      - sediment: suspended_sediment, bed_load, turbidity
      - derived: do_deficit, thermal_stress_index, combined_stress
      
    Returns aligned DataFrame ready for ML.
    '''

def download_and_cache_data(
    river_name: str,
    start_date: str,
    end_date: str
) -> Path:
    '''Download data once, cache locally for reproducibility.'''
```

For your initial PoC, we'll use a Japanese river like:
  - Tone River (Kanto region, well-documented)
  - Shinano River (Chubu region)
  - Yodo River (Kansai region, near IGES)

Recommendation: Start with Yodo River (IGES is in Kanagawa, Yodo is accessible, data available).

PART 3: DATA PREPROCESSOR

Create src/aquanexus/data/preprocessor.py:

```python
class EnvironmentalPreprocessor:
    def __init__(self, config: Settings):
        self.config = config
        self.scaler = None  # Will fit on training data
        
    def fit(self, data: pd.DataFrame) -> None:
        '''Fit scaler on training data.'''
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        '''Normalize features to [0, 1] or StandardScaler.'''
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        '''Create derived features:
           - Interaction terms: temp × DO, flow × sediment
           - Lagged features: previous step values
           - Aggregated: rolling means (7-day, 30-day)
           - Ratios: DO saturation %, thermal load index
        '''
        
    def handle_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        '''Imputation strategy: forward-fill, interpolation, or drop.'''
        
    def create_train_test_split(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        temporal_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''Temporal split: train on early data, test on later data.
        Ensures no data leakage in time-series context.'''
```

PART 4: DATA VALIDATION

Create src/aquanexus/data/validator.py:

```python
class DataValidator:
    @staticmethod
    def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
        '''Return report:
        - Missing values per column
        - Outlier detection (z-score > 3)
        - Duplicates
        - Expected ranges (e.g., DO 0-14 mg/L, temp -2-40°C)
        '''
        
    @staticmethod
    def validate_hecras_output(data: pd.DataFrame) -> bool:
        '''Check HEC-RAS output is valid (no NaNs, reasonable values).'''
        
    @staticmethod
    def validate_temporal_alignment(
        hecras_data: pd.DataFrame,
        monitoring_data: pd.DataFrame
    ) -> bool:
        '''Ensure timestamps align or can be interpolated.'''
```

EXPECTED OUTPUTS FOR PHASE 1B:

✓ src/aquanexus/hecras/reader.py — Reads .ras files, extracts metadata
✓ src/aquanexus/hecras/runner.py — Runs HEC-RAS simulations programmatically
✓ src/aquanexus/data/loader.py — Loads Japanese river data (initially mock, will be real)
✓ src/aquanexus/data/preprocessor.py — Merges HEC-RAS + monitoring data, feature engineering
✓ src/aquanexus/data/validator.py — Data quality checks
✓ requirements.txt updated with: pandas, numpy, xarray, pydantic
✓ test_hecras.py with basic tests
✓ test_data.py with basic tests
✓ Jupyter notebook: 01_hecras_workflow.ipynb — Demo reading .ras file and running simulation
✓ Jupyter notebook: 02_data_exploration.ipynb — Show Japanese river data loading & merging
✓ docs/HECRAS_GUIDE.md — Step-by-step how to set up & run HEC-RAS with AquaNexus
✓ docs/DATA_SOURCES.md — List of Japanese river data sources & how to access

ACTION:
1. Implement HEC-RAS reader/runner
2. Implement Japanese river data loader (start with mock data mimicking real structure)
3. Implement preprocessor and validator
4. Write 2 Jupyter notebooks demonstrating workflow
5. Write 2 docs
6. Write comprehensive unit tests

Show me: (1) All Python files created, (2) Example output from reader.py, (3) Example DataFrame from loader.py, (4) Jupyter notebook preview.
```

End with: "Execute this completely. I need working HEC-RAS integration and data loading by end of this prompt."

---

### Phase 1c: Initial Data Exploration & Validation (Days 4-5)

**Prompt for Claude:**

```
Phase 1b is complete. Now validate end-to-end data pipeline and run exploratory analysis.

PHASE 1C: DATA EXPLORATION & PIPELINE VALIDATION

OBJECTIVE:
1. Load real Japanese river data + HEC-RAS outputs
2. Merge into environmental state vectors
3. Perform exploratory data analysis
4. Identify data gaps & quality issues
5. Finalize feature set for ML

TASKS:

1. Run full pipeline:
   - Download/load Japanese river data (Yodo River recommended)
   - Run HEC-RAS simulation on sample scenario
   - Merge outputs into environmental state DataFrame
   - Validate data integrity

2. Exploratory analysis (Jupyter notebook):
   - Temporal coverage: date range, gaps
   - Feature distributions: histograms, box plots
   - Correlations: heatmap of all environmental variables
   - Temporal patterns: seasonal trends, daily cycles
   - Outliers: detect & document
   - Missing data: patterns and imputation strategy

3. Feature engineering validation:
   - Verify derived features (temp × DO, flow × sediment, etc.)
   - Check for multicollinearity (VIF > 5 indicates redundancy)
   - Confirm interaction terms are meaningful

4. Create data summary report:
   - Row count, date range, completeness %
   - Column descriptions & units
   - Summary statistics
   - Data issues found & how they were handled

EXPECTED OUTPUT:

✓ data/processed/yodo_river_2020_2024.csv — Cleaned, aligned environmental state vectors
✓ Jupyter notebook: 02_data_exploration.ipynb — Full EDA with plots
✓ DEVLOG.md entry: "Phase 1 Complete — Data pipeline working, [N] records, [X]% complete"
✓ docs/DATA_QUALITY_REPORT.md — Issues found, solutions applied

Execute this. Show me: (1) Data summary, (2) Correlation heatmap, (3) Sample output CSV rows.
```

---

## Phase 2: Machine Learning Pipeline (Days 6-12)

**Objective:** Build, train, validate ML model with explainability  
**Output:** Trained model, validation metrics, SHAP explanations  

### Phase 2a: Model Architecture & Training (Days 6-8)

**Prompt for Claude:**

```
Phase 1 complete: data pipeline validated, environmental state vectors ready.

PHASE 2A: MACHINE LEARNING PIPELINE

OBJECTIVE:
1. Build modular ML pipeline (multiple model options)
2. Implement training loop with proper validation
3. Add explainability layer (SHAP, feature importance)
4. Validate model generalization (temporal, spatial splits)

ARCHITECTURE:

Target variable (ecological stress):
  Option A: Binary classification — "healthy" vs "degraded" (threshold-based on DO, temp, sediment)
  Option B: Regression — Habitat suitability score (0–1)
  Option C: Multi-class — Stress type classification (thermal, oxygen, sediment, multi-stressor)

Recommendation: Start with Option B (regression: habitat suitability) → easy to interpret, continuous output

Models to implement:
  1. XGBoost (gradient boosting) — fast, good performance
  2. Random Forest — baseline, good explainability
  3. LSTM (optional, for time-series) — captures temporal patterns

Features to use:
  - Hydrodynamic: depth, velocity, flow, shear_stress, froude_number
  - Thermal: temperature, thermal_gradient
  - Water quality: dissolved_oxygen, do_deficit, cbod, nutrients
  - Sediment: suspended_sediment, bed_load, turbidity
  - Interactions: temp × do, flow × sediment, discharge × temperature
  - Lagged: previous hour values

IMPLEMENTATION:

Create src/aquanexus/ml/models.py:

```python
from dataclasses import dataclass
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import numpy as np

@dataclass
class ModelConfig:
    model_type: str  # 'xgboost', 'random_forest', 'lstm'
    target_type: str  # 'regression', 'classification'
    random_seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1

class ModelFactory:
    @staticmethod
    def create_model(config: ModelConfig):
        if config.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.random_seed,
                objective='reg:squarederror',
                eval_metric='rmse'
            )
        elif config.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=config.random_seed,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

class HabitatPredictor:
    '''Main model wrapper.'''
    
    def __init__(self, model, config: ModelConfig):
        self.model = model
        self.config = config
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data=None):
        '''Train model with optional validation set.'''
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Predict habitat suitability scores [0, 1].'''
        
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Return predictions + confidence intervals.'''
```

Create src/aquanexus/ml/trainer.py:

```python
class ModelTrainer:
    def __init__(self, config: Settings):
        self.config = config
        self.history = {}
        
    def create_temporal_splits(
        self,
        data: pd.DataFrame,
        val_split_date: str,
        test_split_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''Temporal split for time-series:
           - Train: before val_split_date
           - Validation: val_split_date to test_split_date
           - Test: after test_split_date
        '''
        
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_config: ModelConfig
    ) -> HabitatPredictor:
        '''Train model with validation set for early stopping.'''
        
    def hyperparameter_sweep(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict
    ) -> Dict:
        '''Optional: GridSearchCV or RandomizedSearchCV.'''
        
    def save_model(self, model: HabitatPredictor, path: Path):
        '''Serialize trained model (pickle, joblib, or ONNX).'''
        
    def load_model(self, path: Path) -> HabitatPredictor:
        '''Deserialize model.'''
```

EXPECTED OUTPUTS:

✓ src/aquanexus/ml/models.py — Model factory & wrapper
✓ src/aquanexus/ml/trainer.py — Training pipeline
✓ data/models/xgboost_v1.joblib — Trained XGBoost model
✓ Jupyter: 03_model_training.ipynb — Training & validation results
✓ Training metrics logged: RMSE, MAE, R², correlation with actual (if available)
✓ Trained model can predict on new data: y_pred = model.predict(X_test)

Execute: (1) Implement models, (2) Train XGBoost on temporal split, (3) Show validation results.
```

---

### Phase 2b: Explainability & Threshold Detection (Days 9-10)

**Prompt for Claude:**

```
Phase 2a: Model trained and validated. Now add explainability layer.

PHASE 2B: EXPLAINABILITY & THRESHOLD DISCOVERY

OBJECTIVE:
1. Use SHAP to explain model predictions
2. Identify feature importance
3. Discover stress thresholds (critical environmental conditions)
4. Analyze stressor interactions

IMPLEMENTATION:

Create src/aquanexus/ml/explainer.py:

```python
import shap
import matplotlib.pyplot as plt

class HabitatExplainer:
    def __init__(self, model: HabitatPredictor):
        self.model = model
        self.explainer = None
        self.shap_values = None
        
    def fit_explainer(self, X_background: np.ndarray):
        '''Create SHAP explainer using background data.
           TreeExplainer for tree models, KernelExplainer for others.
        '''
        
    def explain_prediction(self, x: np.ndarray) -> Dict:
        '''Explain single prediction:
           - SHAP values (contribution of each feature)
           - Expected value
           - Prediction
           - Top contributing features
        '''
        
    def feature_importance(self) -> pd.DataFrame:
        '''Return feature importance ranked by mean |SHAP|.'''
        
    def interaction_analysis(self, feature1: str, feature2: str) -> Dict:
        '''Detect if two features interact.
           E.g., high temp + low DO is more harmful than sum of individual effects.
        '''
        
    def threshold_discovery(self, data: pd.DataFrame) -> Dict:
        '''Identify critical values:
           - For each feature, find value where prediction changes significantly
           - E.g., DO < 2 mg/L → severe stress
           - Temp > 28°C + DO < 4 mg/L → critical combo
        '''
        
    def plot_force_plot(self, x: np.ndarray, save_path: Path = None):
        '''SHAP force plot showing feature contributions.'''
        
    def plot_summary(self, save_path: Path = None):
        '''SHAP summary plot (bar chart of feature importance).'''
        
    def plot_dependence(self, feature: str, save_path: Path = None):
        '''SHAP dependence plot for a single feature.'''
```

Create src/aquanexus/ml/interactions.py:

```python
class InteractionAnalyzer:
    def __init__(self, model: HabitatPredictor, data: pd.DataFrame):
        self.model = model
        self.data = data
        
    def analyze_pairwise_interactions(self, feature1: str, feature2: str) -> Dict:
        '''For all combinations of feature1 & feature2 values,
           predict habitat suitability. Visualize 2D heatmap.
           
           Example output:
           {
               'feature1': 'temperature',
               'feature2': 'dissolved_oxygen',
               'interaction_strength': 0.45,  # 0–1, higher = stronger interaction
               'critical_zone': {'temp_range': [25, 32], 'do_range': [0, 3]},
               'heatmap': np.array([[...], [...]])
           }
        '''
        
    def threshold_matrix(self) -> pd.DataFrame:
        '''Return table of critical thresholds for each feature & combo.'''
```

EXPECTED OUTPUTS:

✓ src/aquanexus/ml/explainer.py — SHAP-based explainability
✓ src/aquanexus/ml/interactions.py — Interaction analysis
✓ Jupyter: 04_explainability.ipynb — SHAP plots, thresholds, interactions
✓ docs/ML_METHODOLOGY.md — Updated with explainability approach

OUTPUTS FROM EXPLAINABILITY:

- Feature importance ranking: e.g., [DO (0.35), Temperature (0.28), Flow (0.15), ...]
- Critical thresholds: e.g., "DO < 2 mg/L = severe stress", "Temp > 30°C + DO < 4 = critical"
- Interaction heatmaps: temperature × DO, flow × sediment
- Example SHAP explanations: "Prediction of habitat suitability = 0.62 because 
                              (1) temperature is elevated (+0.15), 
                              (2) DO is low (-0.25), 
                              (3) flow is moderate (+0.08), ..."

Execute: (1) Implement SHAP explainer, (2) Run on test set, (3) Generate SHAP plots, 
          (4) Discover thresholds & interactions, (5) Create summary report.
```

---

### Phase 2c: Model Validation & Baselines (Days 11-12)

**Prompt for Claude:**

```
Phase 2b: Explainability complete. Now rigorous validation.

PHASE 2C: MODEL VALIDATION & COMPARISON

OBJECTIVE:
1. Validate model on independent test set
2. Compare against baselines
3. Test generalization (temporal, spatial, event-based)
4. Document all results

BASELINES:

1. Hydraulic-only model: Predict using only depth, velocity, flow
   (Shows value of adding temp, DO, sediment)

2. Simple statistical model: Linear regression on same features
   (Shows value of non-linear ML)

3. Persistence/naive model: Predict that habitat score = previous day's score
   (Shows value of environmental features vs just time)

VALIDATION STRATEGY:

```python
class ModelValidator:
    def __init__(self, model: HabitatPredictor):
        self.model = model
        self.results = {}
        
    def temporal_validation(
        self,
        data: pd.DataFrame,
        train_end: str,
        val_end: str
    ) -> Dict:
        '''Train on data before train_end, 
           validate on data between train_end & val_end (unseen time period).
           Ensures no data leakage.
        '''
        
    def spatial_validation(
        self,
        data_all_reaches: pd.DataFrame
    ) -> Dict:
        '''Train on reaches A & B, test on reach C (geographically unseen).
           Shows if model generalizes across space.
        '''
        
    def event_based_validation(
        self,
        data: pd.DataFrame,
        event_type: str = 'flood'  # or 'drought'
    ) -> Dict:
        '''Train on normal conditions, test on extreme events.
           Shows model can handle out-of-distribution scenarios.
        '''
        
    def compare_to_baselines(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        '''Return metrics for: hydraulic-only, linear, persistence, ML model.
           Show that ML provides improvement.
        '''
        
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        '''RMSE, MAE, R², Pearson correlation, Nash-Sutcliffe Efficiency.'''
```

METRICS TO REPORT:

- RMSE (root mean square error): lower is better
- MAE (mean absolute error): in same units as target
- R² (coefficient of determination): 0–1, higher is better
- Pearson correlation: how well predictions track reality
- Nash-Sutcliffe Efficiency (NSE): common in hydrology, -∞ to 1
- Skill score vs baselines: (Model RMSE - Baseline RMSE) / Baseline RMSE

EXPECTED OUTPUTS:

✓ src/aquanexus/ml/evaluator.py — Validation functions
✓ Validation results table:
  
  Model          | RMSE  | MAE  | R²   | Temporal Gen. | Spatial Gen.
  --------------|-------|------|------|---------------|------ -------
  XGBoost       | 0.12  | 0.08 | 0.84 | ✓ Pass        | ✓ Pass
  Random Forest | 0.15  | 0.10 | 0.81 | ✓ Pass        | ✓ Pass
  Linear Reg.   | 0.25  | 0.18 | 0.62 | ✗ Fail        | ✗ Fail
  Hydraulic-only| 0.30  | 0.22 | 0.55 | ✗ Fail        | ✗ Fail

✓ Jupyter: 05_model_validation.ipynb — Full validation report

DEVLOG.md Update:
  "Phase 2 Complete — Model trained (XGBoost), validated (R²=0.84, RMSE=0.12), 
   explainability added (SHAP), critical thresholds identified. Ready for API/frontend."

Execute: (1) Implement validator, (2) Run all validation splits, (3) Compare baselines,
          (4) Generate validation report with tables & plots.
```

---

## Phase 3: Backend API (Days 13-16)

**Objective:** FastAPI backend with prediction, scenario, explanation endpoints  

### Phase 3a: API Design & Core Endpoints (Days 13-14)

**Prompt for Claude:**

```
Phase 2 complete: trained, validated model with explainability.
Now build FastAPI backend for deployment.

PHASE 3A: FASTAPI BACKEND DESIGN

OBJECTIVE:
1. Design RESTful API with Pydantic schemas
2. Implement core prediction endpoints
3. Add scenario simulation endpoint
4. Add explanation endpoint

ENDPOINTS DESIGN:

POST /predict
  Input: Environmental state (temp, DO, flow, sediment, etc.)
  Output: Habitat suitability score (0–1) + confidence
  Example:
    {
      "temperature": 24.5,
      "dissolved_oxygen": 7.2,
      "flow": 150.0,
      "suspended_sediment": 45.3,
      "depth": 1.8,
      "velocity": 0.85
    }
    →
    {
      "prediction": 0.72,
      "confidence": 0.92,
      "message": "Habitat suitability is moderate"
    }

POST /batch_predict
  Input: CSV file or JSON array of multiple observations
  Output: JSON array of predictions

POST /scenario_run
  Input: Modify specific variables, run HEC-RAS simulation, predict outcome
  Example:
    {
      "scenario_name": "reduced_discharge_20_percent",
      "modifications": {
        "discharge": -0.2  # -20% from baseline
      }
    }
    →
    {
      "scenario_result": {...},
      "environmental_state": {...},
      "prediction": 0.78,
      "comparison_to_baseline": "+0.06 (improvement)"
    }

GET /explain/{prediction_id}
  Output: SHAP explanation for a previous prediction
  Example:
    {
      "expected_value": 0.65,
      "prediction": 0.72,
      "feature_contributions": {
        "dissolved_oxygen": +0.12,
        "temperature": -0.05,
        "flow": +0.04,
        ...
      },
      "top_factors": ["dissolved_oxygen", "temperature", "flow"],
      "likely_driver": "Low oxygen due to effluent discharge"
    }

POST /interactive_analysis
  Input: Range of environmental conditions
  Output: 2D heatmap of predictions (e.g., temperature vs DO)
  Useful for understanding model behavior.

GET /health
  Returns API health status

API SCHEMA (Pydantic):

```python
# src/aquanexus/api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import datetime

class EnvironmentalState(BaseModel):
    """Environmental measurements."""
    timestamp: datetime.datetime
    
    # Hydrodynamic
    depth: float = Field(..., ge=0, description="Water depth (m)")
    velocity: float = Field(..., ge=0, description="Flow velocity (m/s)")
    flow: float = Field(..., ge=0, description="Discharge (m³/s)")
    shear_stress: Optional[float] = Field(None, description="Bed shear stress (Pa)")
    
    # Thermal
    temperature: float = Field(..., ge=-2, le=40, description="Water temperature (°C)")
    
    # Water Quality
    dissolved_oxygen: float = Field(..., ge=0, le=14, description="Dissolved oxygen (mg/L)")
    cbod: Optional[float] = Field(None, ge=0, description="CBOD (mg/L)")
    nutrients: Optional[Dict[str, float]] = None  # nitrogen, phosphorus
    
    # Sediment
    suspended_sediment: float = Field(..., ge=0, description="Suspended sediment (mg/L)")
    turbidity: Optional[float] = Field(None, description="Turbidity (NTU)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00",
                "depth": 1.8,
                "velocity": 0.85,
                "flow": 150.0,
                "temperature": 24.5,
                "dissolved_oxygen": 7.2,
                "suspended_sediment": 45.3
            }
        }

class PredictionResponse(BaseModel):
    """Habitat suitability prediction."""
    prediction: float = Field(..., ge=0, le=1, description="Suitability score")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    message: str = Field(..., description="Human-readable interpretation")
    timestamp: datetime.datetime
    
class ExplanationResponse(BaseModel):
    """SHAP explanation."""
    expected_value: float
    prediction: float
    feature_contributions: Dict[str, float]
    top_factors: List[str]
    critical_thresholds: Dict[str, float]
    
class ScenarioRequest(BaseModel):
    """Scenario simulation."""
    scenario_name: str
    baseline_state: EnvironmentalState
    modifications: Dict[str, float]  # e.g., {"discharge": -0.2}
    run_hecras: bool = True
    
class ScenarioResponse(BaseModel):
    """Scenario result."""
    scenario_name: str
    baseline_prediction: float
    scenario_prediction: float
    improvement: float
    explanation: ExplanationResponse
    hecras_output: Optional[Dict] = None
```

FASTAPI APP STRUCTURE:

```python
# src/aquanexus/api/app.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

app = FastAPI(
    title="AquaNexus API",
    description="Physics-informed ML for aquatic ecosystem diagnosis",
    version="0.1.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
@app.on_event("startup")
async def load_models():
    global model, explainer
    model = load_trained_model()
    explainer = HabitatExplainer(model)
    logger.info("Models loaded")

# Health check
@app.get("/health")
def health():
    return {"status": "ok", "service": "AquaNexus API"}

# Include routers
from aquanexus.api.routes import predictions, scenarios, explanations
app.include_router(predictions.router, prefix="/api", tags=["Predictions"])
app.include_router(scenarios.router, prefix="/api", tags=["Scenarios"])
app.include_router(explanations.router, prefix="/api", tags=["Explanations"])
```

ROUTES:

```python
# src/aquanexus/api/routes/predictions.py

from fastapi import APIRouter, File, UploadFile, HTTPException
import pandas as pd

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(state: EnvironmentalState):
    """Single prediction."""
    try:
        X = prepare_features(state.dict())
        pred = model.predict(X)[0]
        conf = model.predict_with_confidence(X)[1][0]
        msg = interpret_prediction(pred)
        return PredictionResponse(
            prediction=pred,
            confidence=conf,
            message=msg,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    """Batch predictions from CSV."""
    try:
        df = pd.read_csv(file.file)
        predictions = []
        for _, row in df.iterrows():
            X = prepare_features(row.to_dict())
            pred = model.predict(X)[0]
            predictions.append({
                "timestamp": row.get("timestamp"),
                "prediction": pred
            })
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

EXPECTED OUTPUTS:

✓ src/aquanexus/api/app.py — FastAPI main app
✓ src/aquanexus/api/schemas.py — Pydantic request/response models
✓ src/aquanexus/api/routes/predictions.py — /predict, /batch_predict
✓ src/aquanexus/api/routes/scenarios.py — /scenario_run
✓ src/aquanexus/api/routes/explanations.py — /explain
✓ src/aquanexus/api/middleware.py — CORS, error handling, logging
✓ API documentation auto-generated at GET /docs (Swagger UI)
✓ requirements.txt updated: fastapi, uvicorn, pydantic

Execute: (1) Implement FastAPI app & all routes, (2) Write Pydantic schemas,
          (3) Test with curl / Postman, (4) Generate Swagger docs.
```

---

### Phase 3b: Integration & Deployment (Days 15-16)

**Prompt for Claude:**

```
Phase 3a: FastAPI endpoints complete. Now integrate with models and prepare deployment.

PHASE 3B: API INTEGRATION & PRODUCTION SETUP

OBJECTIVE:
1. Load trained models at API startup
2. Integrate HEC-RAS scenario runner
3. Add proper error handling & logging
4. Create Docker image for deployment
5. Write API documentation

TASKS:

1. Model loading & caching (src/aquanexus/api/app.py):
   ```python
   from aquanexus.ml.trainer import ModelTrainer
   from aquanexus.ml.explainer import HabitatExplainer
   import pickle
   
   @app.on_event("startup")
   async def startup_event():
       global model, explainer, preprocessor
       model = pickle.load(open("data/models/xgboost_v1.joblib", "rb"))
       explainer = HabitatExplainer(model)
       preprocessor = EnvironmentalPreprocessor()
       logger.info("✓ Models loaded")
   ```

2. Feature preparation (src/aquanexus/api/app.py):
   ```python
   def prepare_features(data: Dict) -> np.ndarray:
       '''Convert API input to model feature vector.'''
       df = pd.DataFrame([data])
       df = preprocessor.transform(df)
       return df[MODEL_FEATURES].values
   ```

3. Scenario runner integration (src/aquanexus/api/routes/scenarios.py):
   ```python
   @router.post("/scenario_run", response_model=ScenarioResponse)
   async def run_scenario(req: ScenarioRequest):
       if req.run_hecras:
           # Call HEC-RAS with modified parameters
           hecras_output = run_hecras_with_mods(
               baseline_state=req.baseline_state,
               modifications=req.modifications
           )
           scenario_state = merge_hecras_output(hecras_output)
       else:
           # Direct perturbation
           scenario_state = apply_modifications(req.baseline_state, req.modifications)
       
       baseline_pred = predict_state(req.baseline_state)
       scenario_pred = predict_state(scenario_state)
       explanation = explainer.explain_prediction(scenario_state.dict())
       
       return ScenarioResponse(
           scenario_name=req.scenario_name,
           baseline_prediction=baseline_pred,
           scenario_prediction=scenario_pred,
           improvement=scenario_pred - baseline_pred,
           explanation=explanation,
           hecras_output=hecras_output
       )
   ```

4. Error handling & logging:
   ```python
   from fastapi.exceptions import RequestValidationError
   from starlette.middleware.base import BaseHTTPMiddleware
   
   class LoggingMiddleware(BaseHTTPMiddleware):
       async def dispatch(self, request, call_next):
           logger.info(f"{request.method} {request.url}")
           response = await call_next(request)
           logger.info(f"→ {response.status_code}")
           return response
   
   @app.exception_handler(RequestValidationError)
   async def validation_exception_handler(request, exc):
       logger.error(f"Validation error: {exc}")
       return JSONResponse(
           status_code=422,
           content={"detail": "Invalid input"}
       )
   ```

5. Docker setup:
   ```dockerfile
   # Dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY src/ src/
   COPY data/models/ data/models/
   
   EXPOSE 8000
   
   CMD ["uvicorn", "aquanexus.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

6. docker-compose.yml for local dev:
   ```yaml
   version: '3.8'
   services:
     api:
       build: .
       ports:
         - "8000:8000"
       volumes:
         - ./data:/app/data
         - ./src:/app/src
       environment:
         - LOG_LEVEL=INFO
       command: uvicorn aquanexus.api.app:app --reload --host 0.0.0.0
   ```

7. API documentation (docs/API_REFERENCE.md):
   - Endpoint descriptions
   - Request/response examples
   - Error codes
   - Rate limiting (if applicable)
   - Authentication (if applicable)

EXPECTED OUTPUTS:

✓ Model loading integrated into FastAPI startup
✓ All endpoints connected to trained model + explainer
✓ HEC-RAS scenario runner integrated
✓ Comprehensive error handling & logging
✓ Dockerfile & docker-compose.yml
✓ docs/API_REFERENCE.md with full documentation
✓ API deployable via: docker-compose up

TESTING:

Test all endpoints:
  - curl -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" -d '{"temperature": 24.5, ...}'
  - curl -X POST http://localhost:8000/api/scenario_run -H "Content-Type: application/json" -d '{"scenario_name": "test", ...}'
  - curl -X GET http://localhost:8000/docs  (Swagger UI)

Execute: (1) Integrate models & HEC-RAS, (2) Create Docker files, 
          (3) Test all endpoints, (4) Write API documentation.
```

---

## Phase 4: Frontend & Deployment (Days 17-28)

**Objective:** React frontend, polish, documentation  

### Phase 4a: React Frontend Setup (Days 17-20)

**Prompt for Claude:**

```
Phase 3: FastAPI backend complete & tested.
Now build React frontend.

PHASE 4A: REACT FRONTEND

OBJECTIVE:
1. Set up React project with TypeScript
2. Build interactive dashboard
3. Create prediction interface
4. Build scenario builder
5. Integrate explanations visualizations

PROJECT STRUCTURE:

frontend/
├── src/
│   ├── components/
│   │   ├── Dashboard.tsx          # Main dashboard layout
│   │   ├── EnvironmentalInput.tsx # Form to input environmental state
│   │   ├── PredictionCard.tsx     # Display prediction result
│   │   ├── ScenarioBuilder.tsx    # Interface to modify conditions
│   │   ├── ExplainabilityPanel.tsx# SHAP visualizations
│   │   ├── InteractionHeatmap.tsx # 2D heatmap (temp vs DO)
│   │   └── Header.tsx
│   ├── pages/
│   │   ├── Home.tsx
│   │   ├── Predict.tsx
│   │   ├── Scenarios.tsx
│   │   ├── Analyze.tsx
│   │   └── About.tsx
│   ├── services/
│   │   └── api.ts                 # Axios client for FastAPI
│   ├── App.tsx
│   ├── App.css
│   └── index.tsx
├── public/
├── package.json
├── tsconfig.json
└── .env.example

COMPONENTS TO BUILD:

1. EnvironmentalInput.tsx
   - Form with sliders for temp, DO, flow, sediment, depth, velocity
   - Real-time validation (highlight invalid ranges)
   - "Predict" button
   - Load example scenarios

2. PredictionCard.tsx
   - Display suitability score (0–1, color-coded: red to green)
   - Confidence %
   - Interpretation message
   - "Explain" button → shows SHAP

3. ExplainabilityPanel.tsx
   - SHAP bar chart (feature contributions)
   - Key drivers highlighted
   - Critical thresholds highlighted
   - "Interact" button → heatmap

4. InteractionHeatmap.tsx
   - 2D heatmap (e.g., temperature × DO)
   - Predictions as heatmap colors
   - Critical zone highlighted
   - Zoom & hover details

5. ScenarioBuilder.tsx
   - Load baseline state
   - Adjust specific variables (discharge -20%, temp +3°C)
   - "Run Scenario" → calls /scenario_run
   - Compare baseline vs scenario side-by-side

EXAMPLE PAGE: Predict.tsx

```typescript
import React, { useState } from 'react';
import { EnvironmentalInput } from '../components/EnvironmentalInput';
import { PredictionCard } from '../components/PredictionCard';
import { ExplainabilityPanel } from '../components/ExplainabilityPanel';
import { api } from '../services/api';

export const Predict: React.FC = () => {
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async (state: EnvironmentalState) => {
    setLoading(true);
    try {
      const res = await api.post('/api/predict', state);
      setPrediction(res.data);
      
      // Get explanation
      const exp = await api.get(`/api/explain/${res.data.id}`);
      setExplanation(exp.data);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Habitat Suitability Prediction</h1>
      <EnvironmentalInput onSubmit={handlePredict} />
      {loading && <p>Loading...</p>}
      {prediction && <PredictionCard prediction={prediction} />}
      {explanation && <ExplainabilityPanel explanation={explanation} />}
    </div>
  );
};
```

API CLIENT (services/api.ts):

```typescript
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000
});

api.interceptors.response.use(
  res => res,
  err => {
    console.error('API error:', err);
    return Promise.reject(err);
  }
);
```

.env.example:
```
REACT_APP_API_URL=http://localhost:8000
```

EXPECTED OUTPUTS:

✓ React project initialized with TypeScript
✓ All components built
✓ API client integrated
✓ Pages: Home, Predict, Scenarios, Analyze, About
✓ Styled with Material-UI or TailwindCSS
✓ Responsive design (mobile + desktop)

Execute: (1) Create React project, (2) Build all components,
          (3) Integrate API client, (4) Test with running FastAPI backend.
```

---

### Phase 4b: Polish, Testing & Deployment (Days 21-28)

**Prompt for Claude:**

```
Phase 4a: Frontend complete. Now polish, test, and deploy.

PHASE 4B: TESTING, POLISH & DEPLOYMENT

OBJECTIVE:
1. Write comprehensive tests (backend + frontend)
2. Polish UI/UX
3. Create production Docker images
4. Deploy to cloud (optional: Heroku, AWS, Google Cloud)
5. Finalize documentation

TASKS:

1. Backend testing (tests/):
   - Unit tests for data pipeline
   - Unit tests for models
   - Integration tests for API endpoints
   - E2E test: load data → train model → predict
   
2. Frontend testing (frontend/src/__tests__/):
   - Component tests (React Testing Library)
   - API service tests (mock axios)
   - Integration tests

3. UI/UX Polish:
   - Color scheme (professional, accessibility)
   - Loading states, error messages
   - Responsive design tested on mobile
   - Keyboard navigation
   - Dark mode (optional)

4. Docker multi-stage build:
   ```dockerfile
   # Dockerfile (multi-stage)
   
   # Stage 1: Build backend
   FROM python:3.11 as backend
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY src/ src/
   COPY data/models/ data/models/
   
   # Stage 2: Build frontend
   FROM node:18 as frontend
   WORKDIR /app
   COPY frontend/package*.json .
   RUN npm ci
   COPY frontend/ .
   RUN npm run build
   
   # Stage 3: Production
   FROM python:3.11-slim
   WORKDIR /app
   COPY --from=backend /app /app
   COPY --from=frontend /app/build /app/frontend/build
   EXPOSE 8000
   CMD ["uvicorn", "aquanexus.api.app:app", "--host", "0.0.0.0"]
   ```

5. CI/CD with GitHub Actions (.github/workflows/ci.yml):
   ```yaml
   name: CI/CD
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - uses: actions/setup-python@v2
           with:
             python-version: 3.11
         - run: pip install -r requirements.txt
         - run: pytest tests/ --cov=src/aquanexus --cov-report=xml
         - uses: codecov/codecov-action@v2
     build:
       needs: test
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - uses: docker/setup-buildx-action@v1
         - uses: docker/build-push-action@v2
           with:
             tags: aquanexus:latest
   ```

6. Documentation finalization:
   - README.md: Overview, setup, usage
   - ARCHITECTURE.md: System design
   - CONTRIBUTING.md: Development guide
   - API_REFERENCE.md: API docs
   - DEPLOYMENT.md: How to deploy
   - LICENSE: MIT or similar

7. Create DEVLOG.md final summary:
   ```markdown
   # AquaNexus Development Log
   
   ## Project Overview
   - **Name**: AquaNexus
   - **Goal**: Proof-of-concept system integrating HEC-RAS + ML for aquatic ecosystem diagnosis
   - **Status**: Complete ✓
   - **Timeline**: 4 weeks
   - **Last Updated**: [Date]
   
   ## Phases Completed
   
   ### Phase 1: Foundation (Days 1-5)
   - ✓ Project setup & architecture
   - ✓ HEC-RAS integration (reader + runner)
   - ✓ Japanese river data pipeline
   - ✓ Data exploration & validation
   - **Output**: data/processed/yodo_river_2020_2024.csv, 50K records
   
   ### Phase 2: ML (Days 6-12)
   - ✓ Model training (XGBoost: R²=0.84)
   - ✓ Explainability (SHAP)
   - ✓ Threshold discovery
   - ✓ Baseline comparisons
   - **Output**: data/models/xgboost_v1.joblib (trained model)
   
   ### Phase 3: API (Days 13-16)
   - ✓ FastAPI endpoints
   - ✓ Pydantic schemas
   - ✓ Model integration
   - ✓ Docker setup
   - **Output**: http://localhost:8000/docs (Swagger UI)
   
   ### Phase 4: Frontend & Deployment (Days 17-28)
   - ✓ React dashboard
   - ✓ Scenario builder
   - ✓ Explainability visualizations
   - ✓ Tests & CI/CD
   - ✓ Production Docker image
   - **Output**: http://localhost:3000 (React app)
   
   ## Key Decisions
   
   1. **Data**: Yodo River (Kansai, Japan) + synthetic biological labels
   2. **Model**: XGBoost (gradient boosting) for habitat suitability regression
   3. **Explainability**: SHAP for feature importance & interactions
   4. **Backend**: FastAPI with async/await
   5. **Frontend**: React + TypeScript + Material-UI
   6. **Deployment**: Docker + docker-compose
   
   ## Model Performance
   
   | Metric | Value |
   |--------|-------|
   | RMSE (test) | 0.12 |
   | MAE (test) | 0.08 |
   | R² (test) | 0.84 |
   | Pearson r | 0.91 |
   | NSE | 0.82 |
   
   ## Critical Thresholds Discovered
   
   - **DO < 2 mg/L**: Severe oxygen stress
   - **Temp > 30°C + DO < 4 mg/L**: Critical combination
   - **Suspended sediment > 200 mg/L**: Significant turbidity impact
   - **Flow < 50 m³/s + all stressors**: Highest risk state
   
   ## Repository Structure
   
   [Show final tree]
   
   ## Deployment
   
   Local: docker-compose up
   Production: [Cloud platform instructions]
   
   ## Next Steps (Beyond PoC)
   
   1. Integrate real biological validation data
   2. Deploy to cloud (AWS, Google Cloud)
   3. Add user authentication
   4. Expand to multiple rivers in Japan
   5. Publish results
   ```

EXPECTED OUTPUTS:

✓ Complete test suite (backend + frontend)
✓ Multi-stage Docker image
✓ GitHub Actions CI/CD pipeline
✓ Fully polished React UI
✓ Comprehensive documentation (README, ARCHITECTURE, API_REFERENCE, DEPLOYMENT)
✓ DEVLOG.md complete with all phases & decisions
✓ Production-ready code on GitHub
✓ Deployable (docker-compose up or cloud deployment)

Execute: (1) Write tests, (2) Polish UI/UX, (3) Create Docker multi-stage,
          (4) Set up GitHub Actions, (5) Finalize all documentation,
          (6) Create final DEVLOG.md summary.
```

---

## Execution Strategy

**How to use this plan:**

1. **Copy each phase prompt** into a new Claude conversation on your laptop
2. **Run it completely** — don't skip steps
3. **After each phase**, update **DEVLOG.md** with results
4. **Commit to GitHub** after each phase
5. **Test end-to-end** after Phases 2, 3, 4

**Timeline:**
- Days 1-5: Phase 1 (Foundation)
- Days 6-12: Phase 2 (ML)
- Days 13-16: Phase 3 (API)
- Days 17-28: Phase 4 (Frontend & Deployment)

**Success Criteria:**
- ✓ GitHub repo with clean code & documentation
- ✓ Trained model (R² ≥ 0.80)
- ✓ Working FastAPI backend (all endpoints)
- ✓ Working React frontend (predict + scenarios + explain)
- ✓ Docker deployment working
- ✓ Comprehensive DEVLOG.md
- ✓ Presentable to IGES professors & collaborators

---

## Key Files to Track

After completing all phases:

1. **src/aquanexus/** — Core Python package
2. **frontend/src/** — React app
3. **data/models/xgboost_v1.joblib** — Trained model
4. **data/processed/yodo_river_2020_2024.csv** — Processed data
5. **Dockerfile** & **docker-compose.yml** — Deployment
6. **DEVLOG.md** — Development chronicle
7. **README.md** — GitHub front page
8. **docs/** — Architecture, API, deployment guides

This plan is your roadmap. Execute it sequentially. When you hit blockers, come back here with the phase number & issue.


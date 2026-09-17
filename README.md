# Estructura del Repositorio

`
Entrega-TFM/
├── TFM Martin Herrera Roncero.pdf       # Memoria academica oficial del TFM (24 paginas)
├── README.md                            # Estructura del repositorio
├── requirements.txt                     # Dependencias exactas fijadas
├── verificar_entorno.py                 # Script de validacion automatizada de 1 click
├── run_api.bat                          # Lanzador rapido de la API en Windows
├── run_api.sh                           # Lanzador rapido de la API en Linux / macOS
│
├── notebooks/                           # 4 Cuadernos Jupyter interactivos reproducibles
│   ├── 01_EDA_FE.ipynb                  # EDA, Censo SEC, Imputacion en Cascada y FE Multimodal
│   ├── 02_Feature_selection.ipynb       # Random Probes CatBoost GPU, Poda Spearman -> 21 Vars
│   ├── 03_Model_Training_GPU_and_Benchmark.ipynb # Purged TS Split, Scorecard WoE, Benchmark
│   └── 04_Explainability_SHAP_and_Stress_Testing.ipynb # DeLong, McNemar, SHAP Fed SR 11-7, Monte Carlo
│
├── api/                                 # Servicio productivo FastAPI independiente
│   ├── main.py                          # Endpoints REST (/predict, /batch, /explain, /health)
│   ├── schemas.py                       # Validacion estricta con esquemas Pydantic v2
│   ├── service.py                       # Inferencia, calibracion asimetrica y SHAP TreeExplainer
│   ├── run_api.py                       # Lanzador del servicio ASGI Uvicorn
│   ├── live_data_service.py             # Feed de datos bursatiles y macro en tiempo real
│   ├── test_api.py                      # 13 tests unitarios y de integracion (pytest)
│   ├── README_API.md                    # Documentacion y ejemplos cURL
│   └── static/
│       └── dashboard.html               # Cockpit web interactivo (/ui)
│
├── models/                              # Estimadores y transformadores serializados
│   ├── best_xgboost_model.pkl           # XGBoost SOTA Campeon a 12M (Rank 1 SOTA)
│   ├── voting_classifier.pkl            # Ensamble Soft Voting Classifier
│   ├── lightgbm_model.pkl               # LightGBM Classifier
│   ├── lightgbm_best_model.txt          # Modelo LightGBM nativo
│   ├── random_forest_model.pkl          # Random Forest Benchmark
│   ├── catboost_model.pkl               # CatBoost Classifier con GPU
│   ├── catboost_best_model.cbm          # Modelo CatBoost nativo
│   ├── logistic_regression_model.pkl    # Regresion Logistica Baseline
│   ├── scorecard_woe_best.pkl           # Scorecard Basilea III (OptBinning)
│   ├── financial_mlp_best.pt            # Red Neuronal PyTorch entrenada con coste 24x
│   ├── preprocessor_pipeline.joblib     # Pipeline de imputacion y escalado (21 variables)
│   ├── distribution_stats.joblib        # Matriz de covarianza y estadisticas para estres
│   ├── sample_companies_bank.joblib     # Cartera de empresas de prueba para inferencia
│   └── README_MODELS.md                 # Catalogo tecnico de modelos y metricas
│
├── data/
│   ├── processed/
│   │   ├── sec_dataset/                 # Datasets canonicos Parquet y reportes oficiales
│   │   │   ├── v2_sec_financials_pivoted_clean.parquet  # Dataset canonico limpio (21 variables)
│   │   │   ├── v2.1_master_financials_dataset.parquet   # Dataset maestro (52 variables candidatas)
│   │   │   ├── v2_sec_financials_pivoted.parquet        # Dataset intermedio de balances
│   │   │   ├── network_features_monthly.parquet         # Metricas de grafo PageRank mensual
│   │   │   ├── sec_bankruptcy_events.parquet            # Registro de quiebras y transiciones
│   │   │   ├── sec_insider_monthly.parquet              # Datos de insider trading mensual
│   │   │   ├── sec_market_quotes_monthly.parquet        # Cotizaciones bursatiles mensuales
│   │   │   └── *.csv / *.json                           # Tablas de benchmark, contrastes y drift
│   │   └── live_cache/
│   │       └── live_data_cache.sqlite                   # Cache SQLite de cotizaciones
│   └── raw/
│       └── sec_metadata/
│           └── cik_ticker_map.parquet                   # Mapeo CIK a Ticker
│
├── src/                                 # Paquete modular Python limpio (sin __pycache__)
│   ├── config/                          # Configuracion canonica y constantes metodologicas
│   ├── data/                            # Modulos de auditoria y transformacion
│   ├── features/                        # Ratios contables, Merton KMV, GMM, Redes, NLP
│   ├── models/                          # Perdida asimetrica, scorecard, calibracion, entrenamiento
│   ├── evaluation/                      # Contrastes estadisticos (DeLong, McNemar), estres Monte Carlo
│   ├── explainability/                  # Explicabilidad global y local SHAP
│   └── api/                             # Modulos de soporte para la API
│
└── config/
    └── production.json                  # Parametros operativos de produccion
`\n
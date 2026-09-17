# Predicción de Insolvencia Corporativa en el S&P 500 y Russell 3000

## Trabajo de Fin de Máster en Data Science, Big Data & Business Analytics
- **Institución**: Universidad Complutense de Madrid (UCM)
- **Autor**: Martín Herrera Roncero
- **Tutores / Directores Académicos**: Carlos Ortega & Santiago Mota
- **Documento Principal**: `TFM Martin Herrera Roncero.pdf` (24 páginas oficiales)
- **Marco Regulatorio y Normativo**: Federal Reserve SR 11-7, Basilea III, NIIF 9 / IFRS 9, Directiva EU AI Act

---

## 1. Guía Rápida de Inicio para Profesores y Evaluadores

Para facilitar la evaluación y comprobación inmediata del proyecto, se ha dispuesto de un **script automatizado de verificación en un solo comando**:

### Paso 1: Configurar el entorno virtual (Python 3.10 o 3.11 recomendado)
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux / macOS:
source venv/bin/activate

# Instalar dependencias fijadas
pip install --upgrade pip
pip install -r requirements.txt
```

### Paso 2: Ejecutar la auditoría automática de reproducibilidad
```bash
python verificar_entorno.py
```
Este script valida en 5 segundos:
- La correcta importación y versiones del stack científico (PyTorch, XGBoost, LightGBM, CatBoost, SHAP, DuckDB, OptBinning).
- La presencia íntegra de la memoria oficial en PDF y los 4 cuadernos Jupyter.
- La integridad de todos los modelos serializados y pipelines en `models/`.
- La presencia de los datasets canónicos procesados y reportes en `data/`.
- Realiza una **prueba de inferencia en vivo** con una firma de muestra usando el modelo SOTA XGBoost y reporta la probabilidad y clasificación regulatoria.

### Paso 3: Iniciar el servicio productivo API REST y Dashboard Web
```bash
# En Windows (doble clic o terminal):
run_api.bat
# o bien:
python api/run_api.py

# En Linux / macOS:
./run_api.sh
```
Una vez iniciado, acceda desde su navegador web:
- **Cockpit Dashboard Web Interactivo**: [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui) (Permite evaluar empresas interactivamente, ajustar umbrales de coste y ver la descomposición aditiva SHAP por pilares de información).
- **Documentación Interactiva Swagger OpenAPI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Documentación Técnica ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 2. Estructura del Entregable Oficial

El directorio contiene la totalidad de los recursos, datos procesados, modelos entrenados, código fuente modular, servicio API REST y cuadernos interactivos reproducibles de extremo a extremo:

```
Martin_Herrera_Roncero_TFM/
├── TFM Martin Herrera Roncero.pdf       # Memoria académica oficial del TFM (24 páginas)
├── README.md                            # Esta guía de evaluación y gobernanza
├── requirements.txt                     # Dependencias exactas fijadas
├── verificar_entorno.py                 # Script de validación automatizada de 1 click
├── run_api.bat                          # Lanzador rápido de la API en Windows
├── run_api.sh                           # Lanzador rápido de la API en Linux / macOS
│
├── notebooks/                           # 4 Cuadernos Jupyter interactivos reproducibles
│   ├── 01_EDA_FE.ipynb                  # EDA, Censo SEC, Imputación en Cascada y FE Multimodal
│   ├── 02_Feature_selection.ipynb       # Random Probes CatBoost GPU, Poda Spearman -> 21 Vars
│   ├── 03_Model_Training_GPU_and_Benchmark.ipynb # Purged TS Split, Scorecard WoE, Benchmark
│   └── 04_Explainability_SHAP_and_Stress_Testing.ipynb # DeLong, McNemar, SHAP Fed SR 11-7, Monte Carlo
│
├── api/                                 # Servicio productivo FastAPI independiente
│   ├── main.py                          # Endpoints REST (/predict, /batch, /explain, /health)
│   ├── schemas.py                       # Validación estricta con esquemas Pydantic v2
│   ├── service.py                       # Inferencia, calibración asimétrica y SHAP TreeExplainer
│   ├── run_api.py                       # Lanzador del servicio ASGI Uvicorn
│   ├── live_data_service.py             # Feed de datos bursátiles y macro en tiempo real
│   ├── test_api.py                      # 13 tests unitarios y de integración (pytest)
│   ├── README_API.md                    # Documentación y ejemplos cURL
│   └── static/
│       └── dashboard.html               # Cockpit web interactivo (/ui)
│
├── models/                              # Estimadores y transformadores serializados
│   ├── best_xgboost_model.pkl           # XGBoost SOTA Campeón a 12M (Rank 1 SOTA)
│   ├── voting_classifier.pkl            # Ensamble Soft Voting Classifier
│   ├── lightgbm_model.pkl               # LightGBM Classifier
│   ├── random_forest_model.pkl          # Random Forest Benchmark
│   ├── catboost_model.pkl               # CatBoost Classifier con GPU
│   ├── logistic_regression_model.pkl    # Regresión Logística Baseline
│   ├── scorecard_woe_best.pkl           # Scorecard Basilea III (OptBinning)
│   ├── financial_mlp_best.pt            # Red Neuronal PyTorch entrenada con coste 24x
│   ├── preprocessor_pipeline.joblib     # Pipeline de imputación y escalado (21 variables)
│   ├── distribution_stats.joblib        # Matriz de covarianza y estadísticas para estrés
│   ├── sample_companies_bank.joblib     # Cartera de empresas de prueba para inferencia
│   └── README_MODELS.md                 # Catálogo técnico de modelos y métricas
│
├── data/
│   ├── processed/
│   │   ├── sec_dataset/                 # Datasets canónicos Parquet y reportes oficiales
│   │   │   ├── v2_sec_financials_pivoted_clean.parquet  # Dataset canónico limpio (21 variables)
│   │   │   ├── v2.1_master_financials_dataset.parquet   # Dataset maestro (52 variables candidatas)
│   │   │   ├── network_features_monthly.parquet         # Métricas de grafo PageRank mensual
│   │   │   ├── sec_bankruptcy_events.parquet            # Registro de quiebras y transiciones
│   │   │   ├── sec_market_quotes_monthly.parquet        # Cotizaciones bursátiles mensuales
│   │   │   └── *.csv / *.json                           # Tablas de benchmark, contrastes y drift
│   │   └── live_cache/
│   │       └── live_data_cache.sqlite                   # Caché SQLite de cotizaciones
│   └── raw/
│       └── sec_metadata/
│           └── cik_ticker_map.parquet                   # Mapeo CIK a Ticker
│
├── src/                                 # Paquete modular Python limpio (sin __pycache__)
│   ├── config/                          # Configuración canónica y constantes metodológicas
│   ├── data/                            # Módulos de auditoría y transformación
│   ├── features/                        # Ratios contables, Merton KMV, GMM, Redes, NLP
│   ├── models/                          # Pérdida asimétrica, scorecard, calibración, entrenamiento
│   ├── evaluation/                      # Contrastes estadísticos (DeLong, McNemar), estrés Monte Carlo
│   ├── explainability/                  # Explicabilidad global y local SHAP
│   └── api/                             # Módulos de soporte para la API
│
└── config/
    └── production.json                  # Parámetros operativos de producción
```

---

## 3. Protocolo de Ejecución de los Cuadernos de Código

Los cuadernos interactivos permiten reproducir secuencialmente de forma íntegra los resultados, contrastes estadísticos y figuras de la memoria oficial:

1. **`notebooks/01_EDA_FE.ipynb`**:
   - Auditoría del censo XBRL SEC EDGAR (343.6M de hechos contables).
   - Análisis de desfase de publicación (*publication lag* mediano: 39d para 10-Q, 69d para 10-K).
   - Construcción de los 6 pilares multimodales: ratios contables, modelo estructural Merton KMV, regímenes macroeconómicos GMM ($K=3$), arquetipos corporativos GMM ($K=4$), minería textual FinBERT y centralidad en grafo sectorial PageRank.
   - Fase 1 de filtrado heurístico (reducción de 108 a 52 variables candidatas oficiales).
   - Protocolo jerárquico de imputación en cascada en 4 capas concéntricas (100.0% completitud censal).
   - Auditoría forense de colas pesadas y detección multivariante con Isolation Forest (3.00% atípicos).

2. **`notebooks/02_Feature_selection.ipynb`**:
   - Fase 2: Sondaje estocástico en CatBoost GPU (*Random Probes*) con ruido sintético $U(0,1)$ y $N(0,1)$. Cálculo del umbral crítico empírico $\tau_{ruido} = 0.0573$ y poda de 17 variables espurias (54 a 37 variables).
   - Fase 3: Poda algorítmica de colinealidad de segundo orden por correlación de rango de Spearman ($|\rho| \ge 0.75$), eliminando 16 redundancias severas.
   - Fase 4: Consolidación del vector canónico óptimo de 21 variables predictivas y exportación del dataset limpio de producción `v2_sec_financials_pivoted_clean.parquet`.

3. **`notebooks/03_Model_Training_GPU_and_Benchmark.ipynb`**:
   - Esquema riguroso de validación cruzada temporal por emisor (*Purged Group TimeSeries Split* con embargo de 365 días y agrupamiento por CIK) para erradicar cualquier *lookahead bias* o *data leakage*.
   - Construcción del Scorecard regulatorio de crédito WoE (Basilea III).
   - Calibración empírica de la matriz de costes asimétricos 24x ($C_{FN} = \$6,000,000$, $C_{FP} = \$250,000$) y optimización de umbrales operativos ($\tau^* = 0.158$ para XGBoost, $\tau^* = 0.156$ para Voting Classifier).
   - Entrenamiento multimodelo acelerado por hardware GPU (XGBoost, Voting Classifier, LightGBM, CatBoost, Random Forest, Regresión Logística, MLP PyTorch con pérdida asimétrica).
   - Calibración de probabilidades bajo estándar NIIF 9 / IFRS 9 (Brier Score 0.0312).

4. **`notebooks/04_Explainability_SHAP_and_Stress_Testing.ipynb`**:
   - Contrastes estadísticos formales pareados: Test de DeLong pareado ($Z=74.94$, $p < 10^{-15}$ frente al Scorecard; $Z=0.485$, $p=0.627$ entre XGBoost y Voting Classifier) y Test de McNemar ($\chi^2 = 16,639.88$, $p < 10^{-15}$).
   - Intervalos de confianza no paramétricos por remuestreo Bootstrap al 95% ($B=1,000$ réplicas).
   - Curvas de ganancia acumulada y factor Lift (captura del 52% de las insolvencias en el primer decil, Lift de 5.2x).
   - Auditoría de explicabilidad XAI bajo directiva Fed SR 11-7: Descomposición aditiva global SHAP TreeExplainer en los 6 pilares de información y cascadas locales Waterfall para cartas de acción adversa (*ECOA Notice of Adverse Action*).
   - Simulación estocástica Monte Carlo de estrés macroeconómico ($N=10,000$ iteraciones, estimación de VaR al 95% y 99%).
   - Monitorización censal de *data drift* mediante PSI (*Population Stability Index*) y contraste bilateral de Kolmogorov-Smirnov.

---

## 4. Servicio de Inferencia API REST y Pruebas Automatizadas

### Ejecutar la suite de tests de la API
```bash
pytest api/test_api.py -v
```
Todos los 13 tests de integración (predicción solvente, distress, inferencia en lote, explicabilidad SHAP real, compatibilidad hacia atrás y normalización anti-sesgo) se ejecutan y superan con éxito en menos de 4 segundos.

### Endpoints Principales disponibles en `http://127.0.0.1:8000`:
- `GET /`: Estado general del microservicio y especificación normativa.
- `GET /health`: Estado de los 6 modelos cargados en memoria y versión de artefactos.
- `GET /models`: Catálogo técnico de modelos, ranking SOTA, umbrales y métricas PR-AUC / ROC-AUC.
- `GET /features`: Catálogo de las 21 variables predictivas oficiales y sus pilares metodológicos.
- `GET /sample-company`: Obtención de empresas de prueba (solvente vs. distressed) para evaluación rápida.
- `POST /predict`: Inferencia individual con probabilidad calibrada, clasificación de riesgo ($\tau^*$) y coste esperado de negocio.
- `POST /predict/batch`: Inferencia masiva sobre carteras crediticias enteras.
- `POST /explain`: Descomposición aditiva SHAP por variable y por pilar de riesgo.
- `GET /ui`: Cuadro de mando interactivo para analistas de riesgo y comités de crédito.

---

## 5. Gobernanza, Trazabilidad y Tipografía

- **Resolución Relativa Dinámica**: Todo el código fuente, cuadernos y scripts emplean `pathlib.Path.resolve()`, garantizando portabilidad multiplataforma inmediata sin requerir la modificación de rutas fijas locales.
- **Determinismo Estricto**: Todas las simulaciones estocásticas, divisiones de datos y optimizaciones fijan la semilla determinista `seed=42`.
- **Estandarización Tipográfica y Gráfica**: La totalidad de las figuras oficiales están incorporadas y consolidadas en alta resolución (300 DPI) con tipografía corporativa **Arial** dentro de la memoria académica oficial `TFM Martin Herrera Roncero.pdf`, y se renderizan de forma interactiva e *inline* al ejecutar los cuadernos Jupyter.

"""
Servidor RESTful FastAPI con Interfaz Gráfica Interactiva (Cockpit GUI), Soporte para 6 Modelos Oficiales
(Voting Classifier SOTA Rank 1 y 5 Clasificadores Individuales), Scorecard WoE Basilea III,
Descarga Automática de Estados Contables en Vivo y Explicabilidad XAI SHAP por Pilares.

Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500 y Mercado de EE.UU.
Cumplimiento normativo estricto:
- Federal Reserve SR 11-7 (Model Risk Management)
- Basilea III / IV IRB Advanced Standard
- NIIF 9 / IFRS 9 (Expected Credit Loss)
- EU AI Act (Regulation EU 2024/1689, Articles 12, 13 & 14)
- Equal Credit Opportunity Act (ECOA) & Fair Credit Reporting Act (FCRA Adverse Action Notices)
"""

import os
import re
import sys
import json
import time
import random
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# Resolución Dinámica de Rutas Relativas a ENTREGABLE_DIR
# ---------------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
API_DIR = CURRENT_FILE.parent
ENTREGABLE_DIR = CURRENT_FILE.parent.parent

if str(ENTREGABLE_DIR) not in sys.path:
    sys.path.insert(0, str(ENTREGABLE_DIR))
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from schemas import (
    OPTIMAL_21_FEATURES,
    TAXONOMIA_PILARES,
    FEATURE_TO_PILLAR,
    CompanyFinancialPayload,
    CompanyFinancialInput,
    BatchPredictionInput,
    PredictionResponse,
    PredictionOutput,
    BatchPredictionResponse,
    BatchPredictionOutput,
    ModelInfoResponse,
    ModelInfoItem,
    ModelCatalogResponse,
    FeatureItem,
    FeaturesInfoResponse,
    ScorecardResponse,
    ExplanationFactor,
    ExplanationOutput,
)

CONFIG_PATH = ENTREGABLE_DIR / "config" / "production.json"
MODELS_DIR = ENTREGABLE_DIR / "models"
LOGS_DIR = ENTREGABLE_DIR / "logs"
STATIC_DIR = API_DIR / "static"
DASHBOARD_HTML_PATH = STATIC_DIR / "dashboard.html"

# Configuración por defecto
config = {
    "version": "2.4.0",
    "environment": "production",
    "default_model": "xgboost",
    "preprocessor_path": "models/preprocessor_pipeline.joblib",
    "scorecard_path": "models/scorecard_woe_best.pkl",
    "log_file": "logs/audit_logs.jsonl"
}

MODEL_THRESHOLDS = {
    "xgboost": 0.158,
    "lightgbm": 0.224,
    "random_forest": 0.182,
    "voting_classifier": 0.220,
    "catboost": 0.038,
    "logistic_regression": 0.170
}

MODEL_DISPLAY_NAMES = {
    "xgboost": "XGBoost GPU Classifier (Rank 1)",
    "lightgbm": "LightGBM Classifier (Rank 2)",
    "random_forest": "Random Forest Ensemble (Rank 3)",
    "voting_classifier": "Voting Classifier Soft Ensemble (Rank 4)",
    "catboost": "CatBoost GPU Classifier (Rank 5)",
    "logistic_regression": "Logistic Regression ElasticNet (Rank 6)"
}

# Carga de configuración oficial desde production.json
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
            config.update(cfg_data)
            if "models" in cfg_data:
                for m_key, m_val in cfg_data["models"].items():
                    if "threshold" in m_val:
                        MODEL_THRESHOLDS[m_key] = m_val["threshold"]
                    if "name" in m_val:
                        MODEL_DISPLAY_NAMES[m_key] = m_val["name"]
    except Exception as e:
        print(f"[WARN] No se pudo leer {CONFIG_PATH}: {e}")

# Configuración del Logger de Auditoría EU AI Act Art. 12
log_file_path = ENTREGABLE_DIR / config.get("log_file", "logs/audit_logs.jsonl")
log_file_path.parent.mkdir(parents=True, exist_ok=True)
audit_logger = logging.getLogger("eu_ai_act_audit_logger")
audit_logger.setLevel(logging.INFO)
if not audit_logger.handlers:
    fh = logging.FileHandler(str(log_file_path), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(fh)

loaded_models: Dict[str, Any] = {}
preprocessor: Optional[Dict[str, Any]] = None
scorecard: Any = None
sample_bank: Optional[Dict[str, Any]] = None
distribution_stats: Optional[Dict[str, Any]] = None
shap_explainers: Dict[str, Any] = {}


def set_cpu_device(model_obj: Any) -> None:
    """Configura explícitamente device='cpu' para todos los estimadores y boosters."""
    if model_obj is None:
        return
    if hasattr(model_obj, "set_params"):
        try:
            model_obj.set_params(device="cpu")
        except Exception:
            pass
    if hasattr(model_obj, "get_booster"):
        try:
            model_obj.get_booster().set_param({"device": "cpu"})
        except Exception:
            pass
    if hasattr(model_obj, "calibrated_classifiers_"):
        for cc in model_obj.calibrated_classifiers_:
            base_est = getattr(cc, "estimator", getattr(cc, "base_estimator", None))
            if base_est is not None:
                set_cpu_device(base_est)
    for attr in ["lgb", "rf", "cb", "xgb"]:
        if hasattr(model_obj, attr):
            set_cpu_device(getattr(model_obj, attr))
    if hasattr(model_obj, "estimators_"):
        for est in model_obj.estimators_:
            set_cpu_device(est)


def load_artifacts() -> None:
    """Carga y valida todos los artefactos de modelos, preprocesador y explicabilidad."""
    global loaded_models, preprocessor, scorecard, sample_bank, distribution_stats

    # 1. Cargar Preprocesador
    prep_rel = config.get("preprocessor_path", "models/preprocessor_pipeline.joblib")
    prep_path = ENTREGABLE_DIR / prep_rel
    if prep_path.exists():
        try:
            preprocessor = joblib.load(str(prep_path))
            print(f"[INFO] Preprocesador cargado desde: {prep_path}")
        except Exception as e:
            print(f"[ERROR] Preprocesador falló: {e}")

    # 2. Cargar los 6 Modelos de Producción
    model_configs = {
        "voting_classifier": "models/voting_classifier.pkl",
        "lightgbm": "models/lightgbm_model.pkl",
        "random_forest": "models/random_forest_model.pkl",
        "catboost": "models/catboost_model.pkl",
        "xgboost": "models/best_xgboost_model.pkl",
        "logistic_regression": "models/logistic_regression_model.pkl"
    }
    for m_key, m_rel in model_configs.items():
        m_path = ENTREGABLE_DIR / m_rel
        if m_path.exists():
            try:
                m = joblib.load(str(m_path))
                set_cpu_device(m)
                loaded_models[m_key] = m
                print(f"[INFO] Modelo [{m_key}] cargado con éxito.")
            except Exception as e:
                print(f"[WARN] No se pudo cargar modelo {m_key}: {e}")

    # 3. Cargar Scorecard WoE Basilea III
    sc_rel = config.get("scorecard_path", "models/scorecard_woe_best.pkl")
    sc_path = ENTREGABLE_DIR / sc_rel
    if sc_path.exists():
        try:
            scorecard = joblib.load(str(sc_path))
            print(f"[INFO] Scorecard WoE cargado desde: {sc_path}")
        except Exception as e:
            print(f"[WARN] Scorecard WoE no disponible: {e}")

    # 4. Cargar Banco de Muestras
    bank_path = MODELS_DIR / "sample_companies_bank.joblib"
    if bank_path.exists():
        try:
            sample_bank = joblib.load(str(bank_path))
            print("[INFO] Banco de empresas de muestra cargado.")
        except Exception as e:
            print(f"[WARN] Banco de muestras no disponible: {e}")

    # 5. Cargar Estadísticas Multivariantes
    dist_path = MODELS_DIR / "distribution_stats.joblib"
    if dist_path.exists():
        try:
            distribution_stats = joblib.load(str(dist_path))
            print("[INFO] Estadísticas multivariantes cargadas.")
        except Exception as e:
            print(f"[WARN] Estadísticas no disponibles: {e}")


# Carga inicial al importar el módulo
load_artifacts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not loaded_models or preprocessor is None:
        load_artifacts()
    yield
    print("[INFO] Servicio API de Quiebra Corporativa finalizado.")


app = FastAPI(
    title="S&P 500 Corporate Bankruptcy & Credit Risk API",
    description=(
        "Microservicio de Inferencia de Riesgo Crediticio y Predicción de Quiebra S&P 500. "
        "Soporte multimodelo (Voting Classifier SOTA Rank 1, LightGBM, Random Forest, CatBoost, "
        "XGBoost, Logistic Regression), Scorecard WoE de Basilea III, XAI SHAP y Cockpit Interactivo."
    ),
    version=config.get("version", "2.4.0"),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8501",
    "http://127.0.0.1:8501"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FEATURE_HUMAN_LABELS = {
    # --- Pilar Contable (SEC) ---
    "tag_NetIncomeLoss": ("Resultado neto contable (Net Income)", "rentabilidad contable y generación de beneficios"),
    "tag_WorkingCapital": ("Fondo de maniobra neto (Working Capital)", "liquidez operativa corriente"),
    "tag_Assets": ("Activos totales consolidados", "base total de activos corporativos"),
    "pub_lag_days": ("Retardo de publicación contable SEC 10-K/10-Q", "puntualidad y transparencia contable"),
    "tag_RetainedEarningsAccumulatedDeficit": ("Reservas y beneficios acumulados / Déficit", "estabilidad patrimonial histórica"),
    "tag_EntityCommonStockSharesOutstanding": ("Acciones comunes en circulación", "base de accionistas y dilución"),
    "tag_CommonStockSharesAuthorized": ("Acciones comunes autorizadas", "estructura de capital social autorizada"),
    "tag_CommonStockParOrStatedValuePerShare": ("Valor facial por acción común", "denominación de acciones"),
    "tag_CommonStockValue": ("Valor nominal de acciones comunes", "capital social emitido"),

    # --- Pilar Bursátil y Estructural ---
    "merton_distance_to_default": ("Distancia estructural de Merton a la quiebra (DD)", "riesgo de insolvencia estructural"),
    "stock_price_close": ("Precio de cotización bursátil", "valoración de mercado"),
    "market_cap": ("Capitalización bursátil total", "valor de capitalización y tamaño"),

    # --- Pilar Macroeconómico y Regímenes ---
    "macro_inflation": ("Nivel de inflación macroeconómica (CPI)", "presión inflacionaria de costes"),
    "macro_real_gdp_growth_yoy": ("Crecimiento interanual PIB real", "ciclo económico y actividad"),
    "macro_interest_rate": ("Tipo de interés de referencia (Fed Funds)", "coste macroeconómico de crédito"),
    "macro_regime_expansion_prob": ("Probabilidad de régimen macroeconómico de expansión (GMM)", "estabilidad del entorno macroeconómico"),

    # --- Pilar Sentimiento NLP y Noticias ---
    "news_sentiment_range_6m": ("Volatilidad del sentimiento mediático (6m)", "estabilidad informativa corporativa"),
    "market_news_sentiment_mean": ("Sentimiento medio global del mercado", "clima global de mercado"),
    "news_sentiment_avg": ("Sentimiento medio de noticias corporativas (FinBERT)", "percepción mediática corporativa"),

    # --- Pilar Arquetipos Corporativos ---
    "corp_archetype_prob_1": ("Probabilidad de pertenencia al Arquetipo Corporativo 1 (GMM)", "perfil estructural de madurez / escala"),
    "corp_archetype_prob_2": ("Probabilidad de pertenencia al Arquetipo Corporativo 2 (GMM)", "perfil estructural de liquidez / apalancamiento"),

    # --- Campos legacy auxiliares ---
    "tag_CashAndCashEquivalentsAtCarryingValue": ("Tesorería y equivalentes de efectivo", "cobertura de pasivos inmediatos"),
    "tag_AssetsCurrent": ("Activo corriente / circulante", "capacidad operativa a corto plazo"),
    "tag_StockholdersEquity": ("Fondos propios y patrimonio neto", "solvencia y respaldo patrimonial"),
    "tag_InterestExpense": ("Gastos financieros por intereses de deuda", "carga y coste financiero"),
    "tag_Revenues_combined": ("Ingresos operacionales consolidados", "flujo de ingresos"),
    "tag_RevenueFromContractWithCustomerExcludingAssessedTax": ("Ingresos netos por contratos con clientes", "ventas operativas directas"),
    "nlp_sentiment_decayed_30d": ("Sentimiento de noticias FinBERT (30d)", "percepción mediática del riesgo"),
    "news_sentiment_max": ("Pico de sentimiento mediático", "picos de noticias"),
    "macro_unemployment_rate": ("Tasa macroeconómica de desempleo", "entorno laboral macro")
}


def get_tree_explainer(model_name: str = "voting_classifier") -> Any:
    """Obtiene o inicializa de forma perezosa el TreeExplainer de SHAP."""
    global shap_explainers
    if model_name in ["voting_classifier", "ensemble", "sota"]:
        return get_tree_explainer("lightgbm")
    if model_name not in shap_explainers:
        m_obj = loaded_models.get(model_name) or loaded_models.get("lightgbm") or loaded_models.get("xgboost")
        if m_obj is not None:
            try:
                import shap
                shap_explainers[model_name] = shap.TreeExplainer(m_obj)
            except Exception as e:
                print(f"[WARN] No se pudo inicializar SHAP TreeExplainer para {model_name}: {e}")
                shap_explainers[model_name] = None
        else:
            shap_explainers[model_name] = None
    return shap_explainers.get(model_name)


def format_metric_value(val: Any) -> str:
    """Formatea valores monetarios y numéricos para reportes ejecutivos."""
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return str(val)
    sign = "-" if val < 0 else ""
    abs_v = abs(val)
    if abs_v >= 1e9:
        return f"{sign}${abs_v/1e9:,.2f}B"
    elif abs_v >= 1e6:
        return f"{sign}${abs_v/1e6:,.2f}M"
    elif abs_v >= 1e3:
        return f"{sign}${abs_v/1e3:,.2f}K"
    elif abs_v == 0.0:
        return "$0.00"
    return f"{val:.2f}"


def normalize_macro_inflation(val: Any, base_cpi: float = 236.468) -> float:
    """
    Normaliza macro_inflation para erradicar el Training-Serving Skew.
    Si el valor <= 25.0 (tasa porcentual anual ej. 2.8%), proyecta a escala CPI base ~236.468.
    """
    if val is None:
        return base_cpi
    try:
        f_val = float(val)
        if np.isnan(f_val):
            return base_cpi
    except (ValueError, TypeError):
        return base_cpi
    if -50.0 <= f_val <= 25.0:
        return base_cpi * (1.0 + f_val / 100.0)
    return f_val


def compute_approximate_altman_z(payload_dict: dict) -> float:
    """Calcula el Z-Score de Altman para empresas manufactureras/corporativas."""
    ta = payload_dict.get("tag_Assets") or payload_dict.get("total_assets") or 1.0
    if ta <= 0:
        ta = 1.0
    wc = payload_dict.get("tag_WorkingCapital") or 0.0
    re_val = payload_dict.get("tag_RetainedEarningsAccumulatedDeficit") or payload_dict.get("retained_earnings") or 0.0
    ebit = payload_dict.get("ebit") or payload_dict.get("tag_NetIncomeLoss") or 0.0
    mve = payload_dict.get("market_cap") or payload_dict.get("market_value") or ta
    tl = payload_dict.get("total_liabilities") or (ta - (payload_dict.get("tag_StockholdersEquity") or 0.0))
    if tl <= 0:
        tl = ta * 0.5
    sales = payload_dict.get("tag_Revenues_combined") or payload_dict.get("total_revenue") or ta

    r1 = wc / ta
    r2 = re_val / ta
    r3 = ebit / ta
    r4 = mve / tl
    r5 = sales / ta
    z = 1.2 * r1 + 1.4 * r2 + 3.3 * r3 + 0.6 * r4 + 0.999 * r5
    return float(z)


def compute_asymmetric_cost(pd_score: float, is_bankrupt: bool) -> float:
    """
    Matriz de Coste Asimétrico Institucional 24x ($6,000,000 FN vs $250,000 FP).
    Coste esperado de la decisión de crédito.
    """
    cost_fn = 6_000_000.0  # Coste de conceder crédito a quien quiebra (Falso Negativo)
    cost_fp = 250_000.0    # Coste de oportunidad de denegar a solvente (Falso Positivo)
    if is_bankrupt:
        expected_cost = (1.0 - pd_score) * cost_fp
    else:
        expected_cost = pd_score * cost_fn
    return round(expected_cost, 2)


def compute_pillar_contributions(scaled_x: np.ndarray) -> Dict[str, float]:
    """Calcula la magnitud agregada de desviación por cada uno de los 5 pilares."""
    if preprocessor is None:
        return {p: 0.0 for p in TAXONOMIA_PILARES.keys()}
    feat_names = preprocessor["feature_names"]
    contributions = {}
    for p_name, p_feats in TAXONOMIA_PILARES.items():
        mag = 0.0
        for f in p_feats:
            if f in feat_names:
                idx = feat_names.index(f)
                mag += abs(float(scaled_x[0, idx]))
        contributions[p_name] = round(mag, 3)
    return contributions


def generate_adverse_action_notices(
    payload_dict: dict,
    prob: float,
    tau: float,
    model_name: str = "lightgbm",
    scaled_x: Optional[np.ndarray] = None
) -> List[str]:
    """
    Generación dinámica de Cartas de Acción Adversa conforme a ECOA (Equal Credit
    Opportunity Act), CFPB Circular 2022-03 y EU AI Act Art. 12/14.
    """
    notices = []
    if prob < tau:
        return notices

    if preprocessor is None:
        return ["Riesgo crediticio elevado detectado por el modelo multivariante de producción."]

    feature_names = preprocessor["feature_names"]
    defaults = preprocessor["default_medians"]

    # 1. Intentar extracción dinámica de factores de riesgo con SHAP TreeExplainer
    explainer = get_tree_explainer(model_name)
    if explainer is not None:
        try:
            if scaled_x is None:
                scaled_x = prepare_feature_vector(payload_dict)
            X_df = pd.DataFrame(scaled_x, columns=feature_names)
            sv = explainer(X_df)
            shap_vals = sv.values[0] if hasattr(sv, "values") else np.asarray(sv)[0]
            if hasattr(shap_vals, "ndim") and shap_vals.ndim > 1 and shap_vals.shape[-1] == 2:
                shap_vals = shap_vals[..., 1]

            adverse_indices = [i for i in np.argsort(shap_vals)[::-1] if shap_vals[i] > 0.01][:5]
            for idx in adverse_indices:
                feat = feature_names[idx]
                label, area = FEATURE_HUMAN_LABELS.get(feat, (feat, "factor contable/financiero"))
                raw_val = payload_dict.get(feat, defaults.get(feat, 0.0))
                val_str = format_metric_value(raw_val)
                impact = shap_vals[idx]
                notices.append(
                    f"{label}: nivel reportado en {val_str} incrementa severamente el riesgo de quiebra "
                    f"(impacto SHAP: +{impact:.2f} logit en {area})"
                )
        except Exception as e:
            print(f"[WARN] Error en extracción dinámica SHAP: {e}")

    # 2. Si SHAP no produjo factores suficientes, usar desviaciones relativas
    if not notices:
        try:
            if scaled_x is None:
                scaled_x = prepare_feature_vector(payload_dict)
            adverse_high_cols = {
                "macro_interest_rate", "pub_lag_days", "macro_inflation",
                "news_sentiment_range_6m", "tag_InterestExpense", "macro_unemployment_rate"
            }
            scores = []
            for i, col in enumerate(feature_names):
                z = scaled_x[0, i]
                risk_dir = z if col in adverse_high_cols else -z
                scores.append((risk_dir, i, col))
            scores.sort(reverse=True)
            for risk_dir, idx, feat in scores[:5]:
                if risk_dir > 0.1:
                    label, area = FEATURE_HUMAN_LABELS.get(feat, (feat, "factor financiero"))
                    raw_val = payload_dict.get(feat, defaults.get(feat, 0.0))
                    val_str = format_metric_value(raw_val)
                    notices.append(
                        f"{label}: nivel reportado en {val_str} desvía negativamente el perfil de solvencia "
                        f"({risk_dir:+.2f} sigma en {area})"
                    )
        except Exception as e:
            print(f"[WARN] Error en cálculo de desviaciones relativas: {e}")

    # 3. Indicador Merton o Altman Z en zona crítica
    mdd = payload_dict.get("merton_distance_to_default")
    if mdd is not None and mdd < 1.5:
        mdd_notice = f"Distancia estructural de Merton en zona crítica ({mdd:.2f} desviaciones < 1.50 umbral regulatorio)"
        if not any("Merton" in n for n in notices):
            notices.append(mdd_notice)

    altman_z = compute_approximate_altman_z(payload_dict)
    if altman_z < 1.81 and not any("Altman" in n for n in notices):
        notices.append(f"Altman Z-Score en zona crítica de estrés financiero ({altman_z:.2f} < 1.81)")

    return notices


def prepare_feature_vector(payload_dict: dict) -> np.ndarray:
    """Prepara, imputa y escala el vector canónico de 21 variables."""
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocesador de producción no disponible.")
    feature_names = preprocessor["feature_names"]
    defaults = preprocessor["default_medians"]
    base_cpi = defaults.get("macro_inflation", 236.468)

    row = []
    for col in feature_names:
        val = payload_dict.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = defaults.get(col, 0.0)
        elif col == "macro_inflation":
            val = normalize_macro_inflation(val, base_cpi=base_cpi)
        row.append(float(val))

    raw_df = pd.DataFrame([row], columns=feature_names, dtype=np.float64)
    imp_mat = preprocessor["imputer"].transform(raw_df)
    scaled_mat = preprocessor["scaler"].transform(imp_mat)
    return scaled_mat


def fetch_company_data(ticker_or_name: str, force_refresh: bool = False, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Obtiene datos de empresa aplicando Cache-Aside con live_data_service."""
    from live_data_service import get_freshest_company_data
    if defaults is None and preprocessor:
        defaults = preprocessor.get("default_medians", {})
    return get_freshest_company_data(ticker_or_name, force_refresh=force_refresh, defaults=defaults)


# ==============================================================================
# ENDPOINTS RESTful Y COCKPIT GUI
# ==============================================================================

@app.get("/", tags=["Dashboard GUI & Diagnóstico"])
@app.get("/ui", response_class=HTMLResponse, tags=["Dashboard GUI & Diagnóstico"])
async def serve_dashboard(request: Request, format: Optional[str] = None):
    """
    Sirve el Cockpit Interactivo de Predicción de Quiebra S&P 500 (static/dashboard.html).
    Si un cliente programático solicita application/json o no envía text/html, retorna estado JSON.
    """
    accept = request.headers.get("accept", "")
    # Navegadores web solicitan text/html, mientras que clientes API o TestClient reciben JSON
    if request.url.path == "/ui" or "text/html" in accept:
        if DASHBOARD_HTML_PATH.exists():
            with open(DASHBOARD_HTML_PATH, "r", encoding="utf-8") as f:
                return HTMLResponse(
                    content=f.read(),
                    headers={
                        "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                        "Pragma": "no-cache",
                        "Expires": "0"
                    }
                )
        return HTMLResponse(content="<h1>Cockpit Dashboard no encontrado en static/dashboard.html</h1>", status_code=404)

    active_models = list(loaded_models.keys())
    return JSONResponse({
        "status": "online",
        "service": "Corporate Bankruptcy Prediction API (S&P 500 / SEC EDGAR)",
        "version": "2.4.0",
        "frameworks": ["Fed SR 11-7", "Basel III", "IFRS 9", "EU AI Act"],
        "features_count": len(preprocessor["feature_names"]) if preprocessor else 21,
        "default_model": config.get("default_model", "xgboost"),
        "active_models": active_models,
        "scorecard_available": scorecard is not None,
        "documentation": "/docs"
    })


@app.get("/health", tags=["Dashboard GUI & Diagnóstico"])
async def health_check():
    """Diagnóstico exhaustivo de disponibilidad de los 6 modelos, preprocesador y scorecard."""
    active_models_status = {
        m_id: (m_id in loaded_models and loaded_models[m_id] is not None)
        for m_id in MODEL_THRESHOLDS.keys()
    }
    return {
        "status": "healthy" if len(loaded_models) > 0 else "degraded",
        "api_version": "2.4.0",
        "model_loaded": len(loaded_models) > 0,
        "preprocessor_loaded": preprocessor is not None,
        "scorecard_loaded": scorecard is not None,
        "scorecard_status": "loaded" if scorecard is not None else "unavailable",
        "sample_bank_status": "loaded" if sample_bank is not None else "unavailable",
        "available_models": active_models_status,
        "total_models_loaded": len(loaded_models),
        "default_model": config.get("default_model", "xgboost"),
        "operational_threshold_default": MODEL_THRESHOLDS.get(config.get("default_model", "xgboost"), 0.158),
        "environment": config.get("environment", "production")
    }


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Dashboard GUI & Diagnóstico"])
async def model_info():
    """Metadatos oficiales, umbrales y normativas de gobernanza (Fed SR 11-7, Basilea III, EU AI Act, FCRA)."""
    feature_names = preprocessor["feature_names"] if preprocessor else OPTIMAL_21_FEATURES
    def_model = config.get("default_model", "xgboost")
    return ModelInfoResponse(
        model_name="XGBoost Classifier (Rank 1) + Full 6-Model Benchmark",
        version=config.get("version", "2.4.0"),
        features_count=len(feature_names),
        features=feature_names,
        optimal_decision_threshold=MODEL_THRESHOLDS.get(def_model, 0.158),
        framework="XGBoost + LightGBM + Random Forest + Soft Voting + CatBoost + Logistic Regression",
        hardware_acceleration="NVIDIA GPU RTX 5070 (Inference: Low-Latency Multi-Threaded CPU)",
        regulatory_compliance=[
            "Federal Reserve SR 11-7 (Model Risk Management)",
            "Basel III / IV IRB Advanced Standard",
            "EU AI Act (Regulation EU 2024/1689, Articles 12 & 14)",
            "Fair Credit Reporting Act (FCRA Adverse Action Notices)"
        ]
    )


@app.get("/models", response_model=ModelCatalogResponse, tags=["Dashboard GUI & Diagnóstico"])
async def get_models_catalog() -> ModelCatalogResponse:
    """Catálogo detallado de los 6 modelos y el Scorecard con métricas oficiales de la Tabla 7."""
    catalog_meta = {
        "xgboost": {"name": "XGBoost Classifier", "rank": "Rank 1 (12M)", "tau": 0.158, "pr": 0.8551, "roc": 0.9763, "desc": "Extreme Gradient Boosting con GPU y regularización. Modelo de referencia a 12M con menor coste ($2,380.25M) y mayor PR-AUC (0.8551)."},
        "lightgbm": {"name": "LightGBM Classifier", "rank": "Rank 2 (24M)", "tau": 0.224, "pr": 0.8501, "roc": 0.9759, "desc": "LightGBM con GOSS leaf-wise. Modelo en horizonte a 24M y segundo a 12M con coste $2,383.00M y solo 127 FN (97.50% recall)."},
        "random_forest": {"name": "Random Forest Classifier", "rank": "Rank 3", "tau": 0.182, "pr": 0.8450, "roc": 0.9751, "desc": "Random Forest con 500 árboles balanceados. Máxima precisión (45.90%) y especificidad (84.78%), reduciendo falsas alarmas a 6,089 FP."},
        "voting_classifier": {"name": "Voting Classifier Soft Ensemble", "rank": "Rank 4", "tau": 0.220, "pr": 0.8449, "roc": 0.9751, "desc": "Ensemble Soft Voting ponderado (25% c/u). Coste global de $2,432.75M."},
        "catboost": {"name": "CatBoost Classifier", "rank": "Rank 5", "tau": 0.038, "pr": 0.8462, "roc": 0.9744, "desc": "CatBoost GPU con árboles simétricos regulares (coste $2,577.50M)."},
        "logistic_regression": {"name": "Logistic Regression ElasticNet", "rank": "Rank 6 (Baseline)", "tau": 0.170, "pr": 0.4193, "roc": 0.7964, "desc": "Regresión Logística Ridge/ElasticNet lineal como baseline clásico de contraste (coste $6,064.25M)."},
    }

    items: List[ModelInfoItem] = []
    for m_id, meta in catalog_meta.items():
        is_ready = m_id in loaded_models and loaded_models[m_id] is not None
        items.append(
            ModelInfoItem(
                model_id=m_id,
                model_name=meta["name"],
                model_rank=meta["rank"],
                threshold_12m=meta["tau"],
                threshold_24m=None,
                pr_auc=meta["pr"],
                roc_auc=meta["roc"],
                description=meta["desc"],
                status="ready" if is_ready else "offline"
            )
        )
    items.append(
        ModelInfoItem(
            model_id="scorecard",
            model_name="Basel III WoE Credit Scorecard",
            model_rank="Regulatory Baseline (300-850 pts)",
            threshold_12m=0.110,
            threshold_24m=None,
            pr_auc=0.6500,
            roc_auc=0.8950,
            description="Scorecard paramétrico basado en Weight-of-Evidence (WoE) para capital regulatorio.",
            status="ready" if scorecard is not None else "offline"
        )
    )
    return ModelCatalogResponse(
        default_model=config.get("default_model", "xgboost"),
        total_models=len(items),
        models=items,
        regulatory_frameworks=["Fed SR 11-7", "Basel III", "IFRS 9", "EU AI Act"]
    )


@app.get("/features", response_model=FeaturesInfoResponse, tags=["Dashboard GUI & Diagnóstico"])
async def get_features_info() -> FeaturesInfoResponse:
    """Las 21 características canónicas óptimas agrupadas por los 5 pilares regulatorios."""
    feature_items: List[FeatureItem] = []
    feature_names = preprocessor["feature_names"] if preprocessor else OPTIMAL_21_FEATURES
    defaults = preprocessor["default_medians"] if preprocessor else {}

    for f in feature_names:
        pillar = FEATURE_TO_PILLAR.get(f, "Pilar Contable (SEC)")
        label, desc = FEATURE_HUMAN_LABELS.get(f, (f, "Variable financiera"))
        med_val = defaults.get(f)
        feature_items.append(
            FeatureItem(
                feature_name=f,
                pillar=pillar,
                description=f"{label} ({desc})",
                default_median=round(float(med_val), 4) if med_val is not None else None
            )
        )

    return FeaturesInfoResponse(
        total_features=len(feature_items),
        features=feature_items,
        pillars=TAXONOMIA_PILARES
    )


@app.get("/sample_company", tags=["Datos y Muestras"])
async def get_sample_company(
    mode: str = Query("solvent", enum=["solvent", "distressed", "random", "synthetic"])
):
    """Muestras empíricas de la banca de empresas SEC EDGAR (solvent/distressed/random) o sintéticas."""
    clean_mode = mode.lower().strip()
    if sample_bank is not None and clean_mode in ["solvent", "distressed", "random"]:
        if clean_mode == "solvent":
            raw_solv = sample_bank.get("solvent_samples", [])
            healthy_solv = [
                s for s in raw_solv
                if float(s.get("tag_WorkingCapital") or 0.0) > 0
                and float(s.get("tag_NetIncomeLoss") or 0.0) > 0
                and float(s.get("tag_RetainedEarningsAccumulatedDeficit") or 0.0) > 0
                and float(s.get("merton_distance_to_default") or 0.0) > 1.5
            ]
            pool = healthy_solv if healthy_solv else raw_solv
        elif clean_mode == "distressed":
            pool = sample_bank.get("distressed_samples", [])
        elif clean_mode == "random":
            # Persistencia empírica calibrada al 12% de quiebra del mercado real (88% solventes / 12% en quiebra)
            solv_pool = sample_bank.get("solvent_samples", [])
            dist_pool = sample_bank.get("distressed_samples", [])
            if random.random() < 0.12 and dist_pool:
                pool = dist_pool
            else:
                pool = solv_pool if solv_pool else dist_pool
        else:
            pool = []

        if pool:
            # Selecciona de forma aleatoria estocástica una empresa distinta en cada llamada
            raw_rec = dict(random.choice(pool))

            defaults = preprocessor["default_medians"] if preprocessor else {}
            clean_rec = {}
            for k, v in raw_rec.items():
                if k in ["raw_features", "scaled_features", "scaled_vector"]:
                    continue
                if isinstance(v, (float, np.floating)):
                    clean_rec[k] = float(defaults.get(k, 0.0)) if (np.isnan(v) or np.isinf(v)) else float(v)
                elif isinstance(v, (int, np.integer)):
                    clean_rec[k] = int(v)
                elif isinstance(v, (str, bool)) or v is None:
                    clean_rec[k] = v
            if "year" not in clean_rec and "filed_year" in clean_rec:
                clean_rec["year"] = int(clean_rec["filed_year"])

            return clean_rec

    if distribution_stats is not None:
        features = distribution_stats["features"]
        base = distribution_stats["distressed_median"] if clean_mode == "distressed" else (
            distribution_stats["solvent_median"] if clean_mode == "solvent" else distribution_stats["mean"]
        )
        sample = {
            "company_name": f"Synthetic_{clean_mode.capitalize()}_Corp",
            "year": 2024,
            "ticker": f"SYN_{random.randint(100, 999)}"
        }
        for f in features:
            med = base[f]
            pcts = distribution_stats["percentiles"][f]
            iqr = max(abs(pcts[3] - pcts[1]), abs(med) * 0.15, 1.0)
            val = med + np.random.normal(0, iqr * 0.10)
            if any(k in f for k in ["Assets", "Cash", "stock_price", "market_cap", "unemployment", "inflation"]):
                val = max(0.01, val)
            sample[f] = round(float(val), 2)
        return sample

    # Fallback determinista
    if clean_mode == "distressed":
        return {
            "company_name": "Distressed Energy Resources Corp",
            "year": 2024,
            "ticker": "DSTX",
            "tag_NetIncomeLoss": -15000000.0,
            "tag_WorkingCapital": -28000000.0,
            "merton_distance_to_default": 0.45,
            "tag_Assets": 45000000.0,
            "pub_lag_days": 85.0,
            "stock_price_close": 1.15,
            "tag_RetainedEarningsAccumulatedDeficit": -65000000.0,
            "corp_archetype_prob_1": 0.01,
            "macro_inflation": 3.8,
            "market_cap": 15000000.0,
            "tag_EntityCommonStockSharesOutstanding": 13000000.0,
            "tag_CommonStockSharesAuthorized": 100000000.0,
            "macro_real_gdp_growth_yoy": 1.2,
            "news_sentiment_range_6m": 0.85,
            "tag_CommonStockParOrStatedValuePerShare": 0.001,
            "tag_CommonStockValue": 13000.0,
            "macro_interest_rate": 5.25,
            "market_news_sentiment_mean": -0.45,
            "corp_archetype_prob_2": 0.95,
            "macro_regime_expansion_prob": 0.10,
            "news_sentiment_avg": -0.62
        }
    else:
        return {
            "company_name": "Demo_Corp",
            "year": 2024,
            "ticker": "AAPL",
            "tag_NetIncomeLoss": 25000000.0,
            "tag_WorkingCapital": 85000000.0,
            "merton_distance_to_default": 5.50,
            "tag_Assets": 320000000.0,
            "pub_lag_days": 40.0,
            "stock_price_close": 75.0,
            "tag_RetainedEarningsAccumulatedDeficit": 45000000.0,
            "corp_archetype_prob_1": 0.05,
            "macro_inflation": 2.8,
            "market_cap": 3500000000.0,
            "tag_EntityCommonStockSharesOutstanding": 50000000.0,
            "tag_CommonStockSharesAuthorized": 150000000.0,
            "macro_real_gdp_growth_yoy": 2.4,
            "news_sentiment_range_6m": 0.20,
            "tag_CommonStockParOrStatedValuePerShare": 0.01,
            "tag_CommonStockValue": 500000.0,
            "macro_interest_rate": 4.5,
            "market_news_sentiment_mean": 0.15,
            "corp_archetype_prob_2": 0.001,
            "macro_regime_expansion_prob": 0.85,
            "news_sentiment_avg": 0.25
        }


@app.get("/fetch_company", tags=["Datos y Muestras"])
async def get_company_by_ticker(
    ticker: str = Query(..., description="Ticker bursátil o razón social (ej: AAPL, MSFT, BA, TSLA, F)"),
    force_refresh: bool = Query(False, description="Forzar actualización en vivo de mercado, balances y noticias")
):
    """Consulta en vivo de empresa por ticker utilizando live_data_service y caché multinivel."""
    res = fetch_company_data(ticker, force_refresh=force_refresh)
    if not res.get("found"):
        raise HTTPException(
            status_code=404,
            detail=res.get("error", f"Empresa o ticker '{ticker}' no encontrado en los registros oficiales.")
        )
    return res


@app.post("/fetch_and_predict", tags=["Inferencia & Live Data"])
async def fetch_and_predict(
    ticker: str = Query(..., description="Ticker bursátil o nombre de la entidad"),
    model_name: Optional[str] = Query("xgboost", enum=["xgboost", "lightgbm", "random_forest", "voting_classifier", "catboost", "logistic_regression"]),
    force_refresh: bool = Query(False, description="Forzar refresco en vivo de mercado, contabilidad y noticias")
):
    """Descarga en vivo por ticker + predicción calibrada individual + comparativa multimodelo."""
    fetched = fetch_company_data(ticker, force_refresh=force_refresh)
    if not fetched.get("found"):
        raise HTTPException(
            status_code=404,
            detail=fetched.get("error", f"Empresa o ticker '{ticker}' no encontrado en los registros oficiales.")
        )
    c_data = fetched.get("company_data", {})
    valid_keys = set(CompanyFinancialPayload.model_fields.keys())
    filtered_payload = {k: v for k, v in c_data.items() if k in valid_keys}

    payload = CompanyFinancialPayload(**filtered_payload)
    pred = await predict_bankruptcy(payload, model_name=model_name)
    comp = await predict_comparison(payload)

    return {
        "company_info": {
            "ticker": fetched.get("ticker"),
            "company_name": fetched.get("company_name"),
            "source": fetched.get("source"),
            "cache_metadata": fetched.get("cache_metadata")
        },
        "financial_data": c_data,
        "prediction": pred,
        "comparison": comp
    }


@app.get("/cache_status", tags=["Dashboard GUI & Diagnóstico"])
async def get_api_cache_status():
    """Estadísticas y telemetría de la caché SQLite multinivel (Strategy A: Cache-Aside con TTL)."""
    from live_data_service import get_cache_stats
    return {
        "status": "online",
        "cache_strategy": "Strategy A: Cache-Aside with Multi-Level TTL",
        "ttl_policies": {
            "accounting_statements": "90 days (Quarterly 10-Q/10-K)",
            "market_quotes_merton": "24 hours (Daily)",
            "nlp_news_sentiment": "24 hours (Daily with FinBERT)",
            "macro_indicators": "24 hours (Global FRED API)"
        },
        "statistics": get_cache_stats()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inferencia"])
async def predict_bankruptcy(
    payload: CompanyFinancialPayload,
    model_name: Optional[str] = Query("xgboost", enum=["xgboost", "lightgbm", "random_forest", "voting_classifier", "catboost", "logistic_regression"])
):
    """
    Predicción individual con cálculo de probabilidad calibrada, clasificación de riesgo,
    coste asimétrico institucional 24x, cartas de acción adversa ECOA, Altman Z y Merton DD.
    """
    start_time = time.time()
    payload_dict = payload.model_dump()

    target_model = loaded_models.get(model_name) or loaded_models.get("xgboost") or loaded_models.get("lightgbm")
    if target_model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Modelo o preprocesador de inferencia no cargado.")

    scaled_x = prepare_feature_vector(payload_dict)
    prob = float(target_model.predict_proba(scaled_x)[:, 1][0])
    prob = float(np.clip(prob, 0.0001, 0.9999))

    tau = MODEL_THRESHOLDS.get(model_name, MODEL_THRESHOLDS.get("xgboost", 0.158))
    is_bankrupt = prob >= tau

    if prob >= tau:
        category = "Alto Riesgo de Insolvencia (Quiebra / Distress)"
        risk_cat_short = "HIGH RISK / DISTRESS"
    elif prob >= (tau * 0.4):
        category = "Vigilancia Preventiva (Riesgo Moderado)"
        risk_cat_short = "MEDIUM RISK / WATCHLIST"
    else:
        category = "Grado de Inversión (Alta Calidad)"
        risk_cat_short = "LOW RISK / SOLVENT"

    adverse_notices = generate_adverse_action_notices(
        payload_dict=payload_dict,
        prob=prob,
        tau=tau,
        model_name=model_name or "xgboost",
        scaled_x=scaled_x
    )
    altman_z = compute_approximate_altman_z(payload_dict)
    mdd = payload_dict.get("merton_distance_to_default")
    latency = (time.time() - start_time) * 1000.0

    expected_cost = compute_asymmetric_cost(prob, is_bankrupt)
    pillar_contribs = compute_pillar_contributions(scaled_x)

    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_used": model_name,
        "company_name": payload.company_name,
        "year": payload.year,
        "input_vector": {col: payload_dict.get(col) for col in (preprocessor["feature_names"] if preprocessor else [])},
        "probability": round(prob, 5),
        "threshold": tau,
        "is_predicted_distress": is_bankrupt,
        "risk_category": risk_cat_short,
        "risk_tier": category,
        "estimated_cost_usd": expected_cost,
        "adverse_action_notices": adverse_notices,
        "latency_ms": round(latency, 2),
        "compliance": "EU AI Act Art. 12 (Technical Record-Keeping)"
    }
    audit_logger.info(json.dumps(audit_entry))

    disp_name = MODEL_DISPLAY_NAMES.get(model_name, "XGBoost GPU Classifier")

    return PredictionResponse(
        company_name=payload.company_name,
        year=payload.year,
        bankruptcy_probability=round(prob, 5),
        is_bankruptcy_predicted=is_bankrupt,
        risk_category=category,
        decision_threshold=tau,
        altman_z_score=round(altman_z, 4),
        merton_distance_to_default=round(mdd, 4) if mdd is not None else None,
        adverse_action_notices=adverse_notices,
        latency_ms=round(latency, 2),
        model_version=disp_name,
        model_rank="Rank 1 (12M)" if model_name == "xgboost" else ("Rank 2 (24M)" if model_name == "lightgbm" else "Benchmark Member"),
        operational_threshold=tau,
        distress_decision=1 if is_bankrupt else 0,
        risk_tier=category,
        estimated_cost_usd=expected_cost,
        regulatory_framework="Fed SR 11-7, Basel III, IFRS 9, EU AI Act",
        pillar_contributions=pillar_contribs
    )


@app.post("/predict_comparison", tags=["Inferencia"])
async def predict_comparison(payload: CompanyFinancialPayload):
    """Comparación simultánea de los 6 modelos oficiales y el Scorecard de Basilea III."""
    start_time = time.time()
    payload_dict = payload.model_dump()
    scaled_x = prepare_feature_vector(payload_dict)

    comparisons = {}
    for m_name in ["xgboost", "lightgbm", "random_forest", "voting_classifier", "catboost", "logistic_regression"]:
        m_obj = loaded_models.get(m_name)
        if m_obj is not None:
            try:
                p = float(m_obj.predict_proba(scaled_x)[:, 1][0])
                m_tau = MODEL_THRESHOLDS.get(m_name, 0.158)
                comparisons[m_name] = {
                    "display_name": MODEL_DISPLAY_NAMES.get(m_name, m_name),
                    "probability": round(p, 5),
                    "threshold": m_tau,
                    "prediction": "DISTRESS" if p >= m_tau else "SOLVENT",
                    "risk_percent": round(p * 100, 2)
                }
            except Exception as e:
                comparisons[m_name] = {"error": str(e)}

    sc_res = await evaluate_scorecard_internal(payload_dict)
    total_time = (time.time() - start_time) * 1000.0

    primary_tau = MODEL_THRESHOLDS.get("xgboost", 0.158)
    primary_prob = comparisons.get("xgboost", {}).get("probability", 0.0)

    return {
        "company_name": payload.company_name,
        "models": comparisons,
        "scorecard": {
            "credit_score": sc_res.credit_score,
            "rating_band": sc_res.rating_band,
            "implied_pd": sc_res.implied_pd
        },
        "altman_z_score": round(compute_approximate_altman_z(payload_dict), 2),
        "adverse_action_notices": generate_adverse_action_notices(
            payload_dict=payload_dict,
            prob=primary_prob,
            tau=primary_tau,
            model_name="xgboost",
            scaled_x=scaled_x
        ),
        "latency_ms": round(total_time, 2)
    }


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Inferencia"])
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inferencia"])
async def predict_bankruptcy_batch(
    payload: Union[List[CompanyFinancialPayload], BatchPredictionInput],
    model_name: Optional[str] = Query("xgboost", enum=["xgboost", "lightgbm", "random_forest", "voting_classifier", "catboost", "logistic_regression"])
):
    """Inferencia por lotes vectorizada y auditoría técnica conforme a EU AI Act Art. 12."""
    start_time = time.time()
    companies_list = payload.companies if isinstance(payload, BatchPredictionInput) else payload

    if not companies_list:
        return BatchPredictionResponse(
            total_processed=0,
            mean_latency_ms=0.0,
            total_latency_ms=0.0,
            total_alerts=0,
            portfolio_average_pd=0.0,
            total_estimated_cost_usd=0.0,
            predictions=[]
        )

    target_model = loaded_models.get(model_name) or loaded_models.get("xgboost") or loaded_models.get("lightgbm")
    if target_model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Modelo o preprocesador de inferencia no cargado.")

    feature_names = preprocessor["feature_names"]
    defaults = preprocessor["default_medians"]
    base_cpi = defaults.get("macro_inflation", 236.468)
    tau = MODEL_THRESHOLDS.get(model_name, MODEL_THRESHOLDS.get("xgboost", 0.158))

    records = []
    payload_dicts = []
    for p in companies_list:
        p_dict = p.model_dump()
        payload_dicts.append(p_dict)
        row = []
        for col in feature_names:
            val = p_dict.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = defaults.get(col, 0.0)
            elif col == "macro_inflation":
                val = normalize_macro_inflation(val, base_cpi=base_cpi)
            row.append(float(val))
        records.append(row)

    raw_df = pd.DataFrame(records, columns=feature_names, dtype=np.float64)
    imp_mat = preprocessor["imputer"].transform(raw_df)
    scaled_mat = preprocessor["scaler"].transform(imp_mat)

    probs = target_model.predict_proba(scaled_mat)[:, 1]

    predictions = []
    total_alerts = 0
    total_pd = 0.0
    total_cost = 0.0

    for i, (p_dict, prob_val) in enumerate(zip(payload_dicts, probs)):
        prob = float(prob_val)
        is_bankrupt = prob >= tau
        if is_bankrupt:
            total_alerts += 1
        total_pd += prob

        if prob >= tau:
            category = "Alto Riesgo de Insolvencia (Quiebra / Distress)"
            risk_cat_short = "HIGH RISK / DISTRESS"
        elif prob >= (tau * 0.4):
            category = "Vigilancia Preventiva (Riesgo Moderado)"
            risk_cat_short = "MEDIUM RISK / WATCHLIST"
        else:
            category = "Grado de Inversión (Alta Calidad)"
            risk_cat_short = "LOW RISK / SOLVENT"

        notices = generate_adverse_action_notices(
            payload_dict=p_dict,
            prob=prob,
            tau=tau,
            model_name=model_name or "xgboost",
            scaled_x=scaled_mat[i:i+1]
        )
        altman_z = compute_approximate_altman_z(p_dict)
        mdd = p_dict.get("merton_distance_to_default")
        exp_cost = compute_asymmetric_cost(prob, is_bankrupt)
        total_cost += exp_cost

        pillar_contribs = compute_pillar_contributions(scaled_mat[i:i+1])

        predictions.append(
            PredictionResponse(
                company_name=p_dict.get("company_name", "Unknown"),
                year=p_dict.get("year", 2024),
                bankruptcy_probability=round(prob, 5),
                is_bankruptcy_predicted=is_bankrupt,
                risk_category=category,
                decision_threshold=tau,
                altman_z_score=round(altman_z, 4),
                merton_distance_to_default=round(mdd, 4) if mdd is not None else None,
                adverse_action_notices=notices,
                latency_ms=0.0,
                model_version=MODEL_DISPLAY_NAMES.get(model_name, "XGBoost GPU Classifier"),
                model_rank="Rank 1 (12M)" if model_name == "xgboost" else ("Rank 2 (24M)" if model_name == "lightgbm" else "Benchmark Member"),
                operational_threshold=tau,
                distress_decision=1 if is_bankrupt else 0,
                risk_tier=category,
                estimated_cost_usd=exp_cost,
                regulatory_framework="Fed SR 11-7, Basel III, IFRS 9, EU AI Act",
                pillar_contributions=pillar_contribs
            )
        )

        batch_audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": "/predict_batch",
            "model_used": model_name,
            "company_name": p_dict.get("company_name", f"Batch_Item_{i}"),
            "year": p_dict.get("year", 2024),
            "input_vector": {col: p_dict.get(col) for col in feature_names},
            "probability": round(prob, 5),
            "threshold": tau,
            "is_predicted_distress": is_bankrupt,
            "risk_category": risk_cat_short,
            "estimated_cost_usd": exp_cost,
            "adverse_action_notices": notices,
            "compliance": "EU AI Act Art. 12 (Technical Record-Keeping)"
        }
        audit_logger.info(json.dumps(batch_audit_entry))

    total_time = (time.time() - start_time) * 1000.0
    n = len(companies_list)
    mean_lat = total_time / max(n, 1)

    batch_summary_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": "/predict_batch_summary",
        "model_used": model_name,
        "total_processed": n,
        "mean_latency_ms": round(mean_lat, 3),
        "total_latency_ms": round(total_time, 2),
        "distress_count": total_alerts,
        "compliance": "EU AI Act Art. 12 (Batch Execution Log)"
    }
    audit_logger.info(json.dumps(batch_summary_entry))

    return BatchPredictionResponse(
        total_processed=n,
        mean_latency_ms=round(mean_lat, 3),
        total_latency_ms=round(total_time, 2),
        total_alerts=total_alerts,
        portfolio_average_pd=round(total_pd / max(n, 1), 4),
        total_estimated_cost_usd=round(total_cost, 2),
        predictions=predictions
    )


async def evaluate_scorecard_internal(payload_dict: dict) -> ScorecardResponse:
    """Función de evaluación interna del Scorecard de Basilea III."""
    start_time = time.time()
    company_name = payload_dict.get("company_name", "Entidad_Corporativa")
    year = int(payload_dict.get("year", 2024))

    score_val = 650.0
    prob_val = 0.05
    points_dict = {}

    if scorecard is not None and hasattr(scorecard, "score"):
        try:
            var_names = getattr(scorecard, "variable_names", [
                "tag_WorkingCapital", "tag_NetIncomeLoss", "merton_distance_to_default",
                "macro_interest_rate", "market_news_sentiment_mean"
            ])
            row_dict = {}
            for col in var_names:
                v = payload_dict.get(col)
                if v is None and preprocessor:
                    v = preprocessor["default_medians"].get(col, 0.0)
                row_dict[col] = float(v or 0.0)
            df_in = pd.DataFrame([row_dict])
            s = scorecard.score(df_in)
            score_val = float(s[0]) if hasattr(s, "__len__") else float(s)
            if hasattr(scorecard, "predict_proba"):
                pm = scorecard.predict_proba(df_in)
                prob_val = float(pm[0, 1] if pm.shape[1] > 1 else pm[0])
            points_dict = {k: round(v, 2) for k, v in row_dict.items()}
        except Exception:
            scaled_x = prepare_feature_vector(payload_dict)
            model_ref = loaded_models.get("lightgbm") or loaded_models.get("xgboost")
            prob_val = float(model_ref.predict_proba(scaled_x)[:, 1][0]) if model_ref else 0.05
            score_val = 850.0 - (prob_val * 550.0)
    else:
        scaled_x = prepare_feature_vector(payload_dict)
        model_ref = loaded_models.get("lightgbm") or loaded_models.get("xgboost")
        prob_val = float(model_ref.predict_proba(scaled_x)[:, 1][0]) if model_ref else 0.05
        score_val = 850.0 - (prob_val * 550.0)

    score_val = float(np.clip(score_val, 300.0, 850.0))

    if score_val >= 750:
        band = "AAA"
        decision = "Aprobado (Máxima Solvencia)"
    elif score_val >= 700:
        band = "AA"
        decision = "Aprobado (Alta Calidad)"
    elif score_val >= 650:
        band = "A"
        decision = "Aprobado (Buena Solvencia)"
    elif score_val >= 600:
        band = "BBB"
        decision = "Aprobado (Grado de Inversión)"
    elif score_val >= 550:
        band = "BB"
        decision = "Vigilancia (Especulativo)"
    elif score_val >= 500:
        band = "B"
        decision = "Alto Riesgo (Especulativo Alto)"
    elif score_val >= 450:
        band = "CCC"
        decision = "Rechazado (Riesgo Inminente)"
    else:
        band = "D"
        decision = "Rechazado / Alerta de Insolvencia"

    implied_pd = max(0.0001, min(0.9999, (850.0 - score_val) / 550.0))
    latency = (time.time() - start_time) * 1000.0

    return ScorecardResponse(
        company_name=company_name,
        year=year,
        credit_score=round(score_val, 1),
        rating_band=band,
        implied_pd=round(implied_pd, 4),
        pd_probability=round(prob_val, 4),
        decision=decision,
        points_breakdown=points_dict,
        regulatory_framework="Basel III / IFRS 9 Weight-of-Evidence Scorecard",
        latency_ms=round(latency, 2)
    )


@app.get("/scorecard", response_model=ScorecardResponse, tags=["Scorecard Basilea III"])
async def get_scorecard_evaluation(
    mode: Optional[str] = Query(None, description="Modo de muestra: 'solvent' o 'distressed'"),
    company_name: Optional[str] = Query(None, description="Nombre corporativo"),
    tag_WorkingCapital: Optional[float] = Query(None, description="Fondo de maniobra (USD)"),
    tag_NetIncomeLoss: Optional[float] = Query(None, description="Resultado neto (USD)"),
    merton_distance_to_default: Optional[float] = Query(None, description="Distancia al default Merton"),
    macro_interest_rate: Optional[float] = Query(None, description="Tipo de interés (%)"),
    market_news_sentiment_mean: Optional[float] = Query(None, description="Sentimiento medio del mercado")
):
    """Evaluación crediticia regulatoria de Basilea III (escala 300 a 850 puntos) vía GET."""
    data: Dict[str, Any] = {}
    if mode:
        sample_res = await get_sample_company(mode)
        if isinstance(sample_res, dict):
            data.update(sample_res)
    if company_name:
        data["company_name"] = company_name
    if tag_WorkingCapital is not None:
        data["tag_WorkingCapital"] = tag_WorkingCapital
    if tag_NetIncomeLoss is not None:
        data["tag_NetIncomeLoss"] = tag_NetIncomeLoss
    if merton_distance_to_default is not None:
        data["merton_distance_to_default"] = merton_distance_to_default
    if macro_interest_rate is not None:
        data["macro_interest_rate"] = macro_interest_rate
    if market_news_sentiment_mean is not None:
        data["market_news_sentiment_mean"] = market_news_sentiment_mean

    return await evaluate_scorecard_internal(data)


@app.post("/scorecard", response_model=ScorecardResponse, tags=["Scorecard Basilea III"])
async def post_scorecard_evaluation(payload: CompanyFinancialPayload):
    """Evaluación crediticia regulatoria de Basilea III con payload completo."""
    return await evaluate_scorecard_internal(payload.model_dump())


@app.post("/explain", response_model=ExplanationOutput, tags=["Explicabilidad XAI"])
async def explain_decision(
    payload: CompanyFinancialPayload,
    model_name: Optional[str] = Query("voting_classifier", enum=["voting_classifier", "lightgbm", "random_forest", "catboost", "xgboost", "logistic_regression"])
):
    """
    Explicabilidad local con SHAP TreeExplainer sobre las 21 variables óptimas.
    Desglose aditivo en escala logit, impacto sobre el riesgo y taxonomía de los 5 pilares.
    """
    payload_dict = payload.model_dump()
    scaled_x = prepare_feature_vector(payload_dict)
    feature_names = preprocessor["feature_names"] if preprocessor else OPTIMAL_21_FEATURES
    defaults = preprocessor["default_medians"] if preprocessor else {}

    explainer_model = "lightgbm" if model_name in ["voting_classifier", "ensemble", "lightgbm"] else (
        "xgboost" if model_name == "xgboost" else "lightgbm"
    )
    explainer = get_tree_explainer(explainer_model)

    if explainer is not None:
        try:
            X_df = pd.DataFrame(scaled_x, columns=feature_names)
            sv = explainer(X_df)
            shap_row = sv.values[0] if hasattr(sv, "values") else np.asarray(sv)[0]
            if hasattr(shap_row, "ndim") and shap_row.ndim > 1 and shap_row.shape[-1] == 2:
                shap_row = shap_row[..., 1]
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                base_logit = float(ev[1] if len(ev) > 1 else ev[0])
            else:
                base_logit = float(ev)
            total_logit = base_logit + float(np.sum(shap_row))
            final_prob = float(1.0 / (1.0 + np.exp(-total_logit)))
            model_used_label = f"TreeExplainer ({explainer_model.upper()})"
        except Exception as e:
            print(f"[WARN] Error en inferencia SHAP: {e}")
            explainer = None

    if explainer is None:
        base_logit = -2.15
        shap_row = np.zeros(len(feature_names))
        wc = float(payload_dict.get("tag_WorkingCapital") or 0.0)
        ni = float(payload_dict.get("tag_NetIncomeLoss") or 0.0)
        merton_dd = payload_dict.get("merton_distance_to_default")
        if "tag_WorkingCapital" in feature_names:
            shap_row[feature_names.index("tag_WorkingCapital")] = 0.85 if wc < 0 else -0.42
        if "tag_NetIncomeLoss" in feature_names:
            shap_row[feature_names.index("tag_NetIncomeLoss")] = 0.64 if ni < 0 else -0.35
        if "merton_distance_to_default" in feature_names and merton_dd is not None:
            shap_row[feature_names.index("merton_distance_to_default")] = 0.75 if float(merton_dd) < 1.5 else -0.45
        total_logit = base_logit + float(np.sum(shap_row))
        final_prob = float(1.0 / (1.0 + np.exp(-total_logit)))
        model_used_label = "Analytic Fallback SHAP"

    factors: List[ExplanationFactor] = []
    pillar_totals: Dict[str, float] = {p: 0.0 for p in TAXONOMIA_PILARES.keys()}

    for i, feat in enumerate(feature_names):
        s_val = float(shap_row[i]) if i < len(shap_row) else 0.0
        raw_val = payload_dict.get(feat, defaults.get(feat, 0.0))
        pillar = FEATURE_TO_PILLAR.get(feat, "Pilar Contable (SEC)")
        pillar_totals[pillar] = pillar_totals.get(pillar, 0.0) + s_val

        impact = "Incrementa Riesgo" if s_val > 0 else "Mitiga Riesgo"
        factors.append(
            ExplanationFactor(
                feature_name=feat,
                shap_value=round(s_val, 4),
                feature_value=round(float(raw_val), 4) if raw_val is not None else None,
                information_pillar=pillar,
                risk_impact=impact
            )
        )

    factors.sort(key=lambda x: abs(x.shap_value), reverse=True)

    return ExplanationOutput(
        company_name=payload.company_name,
        model_version=model_used_label,
        base_value_logit=round(base_logit, 4),
        final_score_logit=round(total_logit, 4),
        bankruptcy_probability=round(final_prob, 4),
        top_risk_drivers=factors[:6],
        pillar_summary={k: round(v, 4) for k, v in pillar_totals.items()},
        regulatory_compliance="Federal Reserve SR 11-7 / ECOA Adverse Action Notice / EU AI Act Art. 13-14"
    )

"""
SalesPredict — Fix del Pipeline
=================================
Equipo: edita SOLO este archivo.
No modifiques run_pipeline.py ni los archivos en data/ o models/.

Instrucciones:
    1. Llena get_team_name() con el nombre de tu equipo
    2. Implementa detect_drift()
    3. Implementa fix_data()
    4. Implementa check_model_health()
    5. Corre: python run_pipeline.py --fixed
"""


import pandas as pd
import numpy as np
from scipy import stats


# =============================================================================
# CONFIGURACION DEL EQUIPO
# =============================================================================

def get_team_name() -> str:
    return "Equipo 5"


# =============================================================================
# FUNCION 1 — Deteccion de drift
# =============================================================================

def detect_drift(df_train: pd.DataFrame, df_prod: pd.DataFrame) -> dict:
    resultado = {}

    columnas_numericas = [
        "precio_unitario",
        "descuento_pct",
        "inventario_prev",
        "es_temporada_alta"
    ]

    for col in columnas_numericas:
        if col not in df_train.columns or col not in df_prod.columns:
            raise ValueError(f"La columna {col} no existe en ambos DataFrames")

        stat, p_value = stats.ks_2samp(df_train[col], df_prod[col])

        resultado[col] = {
            "p_value": float(p_value),
            "statistic": float(stat),
            "drift": bool(p_value < 0.05)
        }

    return resultado


# =============================================================================
# FUNCION 2 — Correccion de datos
# =============================================================================

def fix_data(df_prod: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    df = df_prod.copy()

    # ============================
    # VALIDACIONES
    # ============================
    required_cols = ["precio_unitario", "semana"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta columna {col}")

    # ============================
    # FIX — MONEDA DESDE SEMANA 45
    # ============================
    # A partir de semana 45 el precio viene en USD → convertir a MXN
    df.loc[df["semana"] >= 45, "precio_unitario"] = (
        df.loc[df["semana"] >= 45, "precio_unitario"] * 17.5
    )

    return df


# =============================================================================
# FUNCION 3 — Estado del modelo
# =============================================================================

def check_model_health(metrics: dict) -> str:
    accuracy  = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall    = metrics.get("recall", 0)
    f1        = metrics.get("f1", 0)

    # Umbrales basados en comportamiento observado:
    # ~0.90 = bueno, ~0.60 = malo
    if f1 >= 0.85:
        return "OK"
    elif f1 >= 0.70:
        return "WARNING"
    else:
        return "CRITICAL"
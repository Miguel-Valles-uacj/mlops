"""
SalesPredict — Fix del Pipeline
=================================
Equipo 4
"""

import pandas as pd
import numpy as np
from scipy import stats

def get_team_name() -> str:
    return "Equipo 4"

def detect_drift(df_train: pd.DataFrame, df_prod: pd.DataFrame) -> dict:
    numeric_cols = ["precio_unitario", "descuento_pct", "inventario_prev", "es_temporada_alta"]
    resultado = {}
    for col in numeric_cols:
        train_vals = df_train[col].dropna()
        prod_vals = df_prod[col].dropna()
        statistic, p_value = stats.ks_2samp(train_vals, prod_vals)
        resultado[col] = {
            "p_value": float(p_value),
            "statistic": float(statistic),
            "drift": bool(p_value < 0.05)
        }
    return resultado

def fix_data(df_prod: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    df = df_prod.copy()
    median_prod = df["precio_unitario"].median()
    if median_prod < 300:
        df["precio_unitario"] = df["precio_unitario"] * 17.5
        print("[fix_data] Precios convertidos de USD a MXN")
    mask = df["semana"] >= 45
    if (df.loc[mask, "es_temporada_alta"] == 0).any():
        df.loc[mask, "es_temporada_alta"] = 1
        print("[fix_data] Corregidos flags de temporada alta")
    if df["precio_unitario"].isnull().any():
        raise ValueError("Precios nulos")
    return df

def check_model_health(metrics: dict) -> str:
    f1 = metrics.get("f1", 0)
    if f1 >= 0.85:
        return "OK"
    elif f1 >= 0.70:
        return "WARNING"
    else:
        return "CRITICAL"

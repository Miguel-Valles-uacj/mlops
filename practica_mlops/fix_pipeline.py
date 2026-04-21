"""
SalesPredict - Fix del Pipeline
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
    return "TSP: alejandro, diego y jorge"

# =============================================================================
# FUNCION 1 - Deteccion de drift
# =============================================================================

def detect_drift(df_train: pd.DataFrame, df_prod: pd.DataFrame) -> dict:
    resultado = {}
    columnas_numericas = ["precio_unitario", "descuento_pct", "inventario_prev", "es_temporada_alta"]
    
    for col in columnas_numericas:
        if col in df_train.columns and col in df_prod.columns:
            s_val, p_val = stats.ks_2samp(df_train[col], df_prod[col])
            resultado[col] = {
                "p_value": float(p_val),
                "statistic": float(s_val),
                "drift": bool(p_val < 0.05)
            }
            
    return resultado

# =============================================================================
# FUNCION 2 - Correccion de datos
# =============================================================================

def fix_data(df_prod: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    df = df_prod.copy()

    if df.empty:
        raise ValueError("El DataFrame esta vacio.")

    # Falla 2: precio_unitario llega en USD (semanas 45-52)
    df.loc[df["semana"] >= 45, "precio_unitario"] = df["precio_unitario"] * 17.5
    
    # Falla 1: data drift estacional
    df.loc[(df["semana"] >= 45) & (df["semana"] <= 52), "es_temporada_alta"] = True

    return df

# =============================================================================
# FUNCION 3 - Estado del modelo
# =============================================================================

def check_model_health(metrics: dict) -> str:
    f1 = metrics.get("f1", 0)

    if f1 >= 0.85:
        return "OK"
    elif f1 >= 0.75:
        return "WARNING"
    else:
        return "CRITICAL"
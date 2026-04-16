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
    """
    Retorna el nombre de tu equipo.
    Ejemplo: return "Equipo 3 — Ana, Luis, Diego"
    """
    # TODO: cambia esto con el nombre de tu equipo
    return "Equipo 7 - Raúl, Enrique, Erick"


# =============================================================================
# FUNCION 1 — Deteccion de drift
# =============================================================================

def detect_drift(df_train: pd.DataFrame, df_prod: pd.DataFrame) -> dict:
    resultado = {}

    # seleccionar colum num
    numeric_cols = df_train.select_dtypes(include='number').columns

    for col in numeric_cols:
        if col in df_prod.columns:
            # eliminar valores nan
            train_values = df_train[col].dropna()
            prod_values = df_prod[col].dropna()

            # aplicar test ks
            stat, p_value = stats.ks_2samp(train_values, prod_values)

            resultado[col] = {
                "p_value": float(p_value),
                "statistic": float(stat),
                "drift": p_value < 0.05
            }

    return resultado


# =============================================================================
# FUNCION 2 — Correccion de datos
# =============================================================================

def fix_data(df_prod: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    df = df_prod.copy()

    # revisar que si existan las colum
    required_cols = ["precio_unitario", "descuento_pct", "inventario_prev", "es_temporada_alta", "semana"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida en produccion: {col}")

    # sacar la mediana del precio en train y en prod
    train_median = df_train["precio_unitario"].median()
    prod_median = df["precio_unitario"].median()

    # si el precio de prod es muy abajo prob sea en dolares
    if prod_median < train_median * 0.2:
        tipo_cambio = 17.5
        df["precio_unitario"] = df["precio_unitario"] * tipo_cambio

    # validar que no haya precios en 0 o negativos
    if (df["precio_unitario"] <= 0).any():
        raise ValueError("Se detectaron precios_unitarios inválidos (<= 0)")

    # revisar que las semanas esten en rango normal
    if not df["semana"].between(1, 52).all():
        raise ValueError("Valores de semana fuera de rango (1-52)")

    # corregir la temporada alta
    df["es_temporada_alta"] = df["semana"].between(45, 52).astype(int)

    # revisar que descuento este en rango normal
    if not df["descuento_pct"].between(0, 1).all():
        raise ValueError("descuento_pct fuera de rango [0, 1]")

    # verificar que no haga inventario negativo
    if (df["inventario_prev"] < 0).any():
        raise ValueError("inventario_prev no puede ser negativo")

    return df


# =============================================================================
# FUNCION 3 — Estado del modelo
# =============================================================================

def check_model_health(metrics: dict) -> str:
    accuracy  = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall    = metrics.get("recall", 0)
    f1        = metrics.get("f1", 0)


    for name, value in metrics.items():
        if not (0 <= value <= 1):
            raise ValueError(f"Métrica inválida: {name} debe estar entre 0 y 1")


    # CRITICAL desempeño malo
    if f1 < 0.60 or accuracy < 0.65:
        return "CRITICAL"

    # WARNING desempeño medio
    if f1 < 0.75 or precision < 0.70 or recall < 0.70:
        return "WARNING"

    # OK desempeño aceptable
    return "OK"

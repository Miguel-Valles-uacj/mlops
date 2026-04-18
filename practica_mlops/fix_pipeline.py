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
    return "Equipo 3 — Omar, Villedo, Blanca"


# =============================================================================
# FUNCION 1 — Deteccion de drift
# =============================================================================

def detect_drift(df_train: pd.DataFrame, df_prod: pd.DataFrame) -> dict:
    """
    Detecta data drift comparando la distribucion de cada columna numerica
    entre los datos de entrenamiento y los de produccion.

    Usa el test de Kolmogorov-Smirnov (scipy.stats.ks_2samp) por columna.
    Una columna tiene drift si su p-value es menor a 0.05.

    Parametros:
        df_train: DataFrame con los datos de entrenamiento (columnas numericas)
        df_prod:  DataFrame con los datos de produccion (columnas numericas)

    Retorna:
        dict con este formato por cada columna numerica:
        {
            "nombre_columna": {
                "p_value": float,   # valor p del test KS
                "statistic": float, # estadistico KS
                "drift": bool       # True si p_value < 0.05
            },
            ...
        }
    """
    columnas_numericas = ["precio_unitario", "descuento_pct", "inventario_prev", "es_temporada_alta"]
    resultado = {}

    for col in columnas_numericas:
        if col not in df_train.columns or col not in df_prod.columns:
            raise ValueError(f"La columna '{col}' no existe en uno de los DataFrames.")

        stat, p_value = stats.ks_2samp(df_train[col].dropna(), df_prod[col].dropna())

        resultado[col] = {
            "p_value":   round(float(p_value), 6),
            "statistic": round(float(stat), 6),
            "drift":     bool(p_value < 0.05),
        }

    return resultado


# =============================================================================
# FUNCION 2 — Correccion de datos
# =============================================================================

def fix_data(df_prod: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe los datos de produccion con fallas y retorna un DataFrame corregido.

    Las fallas que debes corregir:
        - Falla 1: data drift estacional (semanas 45-52 = temporada alta)
        - Falla 2: precio_unitario llega en USD en vez de MXN

    Parametros:
        df_prod:  DataFrame de produccion con fallas
        df_train: DataFrame de entrenamiento limpio (puedes usarlo como referencia)

    Retorna:
        DataFrame corregido listo para hacer predicciones

    IMPORTANTE:
        - No modifiques df_prod directamente, trabaja sobre una copia
        - Si detectas datos invalidos, lanza ValueError con un mensaje descriptivo
        - No silencies errores con try/except vacios
    """
    df = df_prod.copy()

    # Validacion basica
    if "precio_unitario" not in df.columns:
        raise ValueError("El DataFrame de produccion no contiene la columna 'precio_unitario'.")
    if "es_temporada_alta" not in df.columns:
        raise ValueError("El DataFrame de produccion no contiene la columna 'es_temporada_alta'.")
    if "semana" not in df.columns:
        raise ValueError("El DataFrame de produccion no contiene la columna 'semana'.")

    # --- Falla 1: precio_unitario en USD en vez de MXN ---
    # Si la mediana de produccion es menor al 30% de la mediana de entrenamiento,
    # los precios estan en dolares y hay que convertirlos a pesos.
    TIPO_CAMBIO = 17.5
    mediana_train = df_train["precio_unitario"].median()
    mediana_prod  = df["precio_unitario"].median()

    if mediana_prod < mediana_train * 0.30:
        df["precio_unitario"] = df["precio_unitario"] * TIPO_CAMBIO

    # --- Falla 2: es_temporada_alta no marcada en semanas 45-52 ---
    # El modelo fue entrenado sin datos de temporada alta (siempre 0),
    # pero en produccion las semanas 45-52 deben tener es_temporada_alta = 1.
    df.loc[df["semana"].between(45, 52), "es_temporada_alta"] = 0

    return df


# =============================================================================
# FUNCION 3 — Estado del modelo
# =============================================================================

def check_model_health(metrics: dict) -> str:
    """
    Evalua el estado del modelo basandose en sus metricas actuales.

    Parametros:
        metrics: dict con keys "accuracy", "precision", "recall", "f1"
                 Cada valor es un float entre 0 y 1.

    Retorna:
        str: "OK", "WARNING" o "CRITICAL"

    Reglas (define TUS umbrales y justificalos en el documento):
        - "OK"       si las metricas estan dentro de rangos aceptables
        - "WARNING"  si alguna metrica empezo a degradarse pero no es critico
        - "CRITICAL" si el modelo esta fallando gravemente

    IMPORTANTE: los umbrales que elijas deben estar justificados en el
    documento con base en las metricas que observaste en F1 y F4.
    """
    accuracy  = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall    = metrics.get("recall", 0)
    f1        = metrics.get("f1", 0)

    # Umbrales basados en las metricas originales del modelo (Accuracy ~0.91, F1 ~0.89)
    # OK      : f1 >= 0.80 y accuracy >= 0.85  → modelo funciona bien
    # WARNING : f1 >= 0.60 o accuracy >= 0.70  → degradacion leve, vigilar
    # CRITICAL: cualquier valor por debajo      → falla grave, intervenir

    if f1 >= 0.80 and accuracy >= 0.85:
        return "OK"
    elif f1 >= 0.60 or accuracy >= 0.70:
        return "WARNING"
    else:
        return "CRITICAL"

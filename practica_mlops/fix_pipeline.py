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
    return "Equipo 9 - Jenni, Jaziel y Eliezer"


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
    # TODO: implementa el test KS para cada columna numerica
    # Pista: usa stats.ks_2samp(df_train[col], df_prod[col])
    # Pista: las columnas numericas son: precio_unitario, descuento_pct,
    #        inventario_prev, es_temporada_alta
    resultado = {}

    columnas_numericas = df_train.select_dtypes(include=[np.number]).columns
    
    for col in columnas_numericas:
        if col in df_prod.columns:
            # Prueba KS de scipy.stats
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

    # Validacion de no tener valores nulos
    if df.isnull().values.any():
        raise ValueError("Los datos de producción contienen valores nulos. No se puede procesar.")

    # Falla 1: Data drift estacional semana 45 a 52
    if 'semana' in df.columns and 'es_temporada_alta' in df.columns:
        # Forzamos que a partir de la semana 45 el flag de temporada alta sea 1
        df.loc[df['semana'] >= 45, 'es_temporada_alta'] = 1

    # Falla 2: precio_unitario en USD en vez de MXN
    if 'precio_unitario' in df.columns:
        umbral_usd = df_train['precio_unitario'].mean() / 5
        # Todo lo qe cueste menos de 100 estan en dolares, lo convertimos a pesos
        mask_usd = df['precio_unitario'] < 100
        
        # Multiplicamos por 17.5 solo las filas que cumplen la condición
        df.loc[mask_usd, 'precio_unitario'] = df.loc[mask_usd, 'precio_unitario'] * 17.5

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
    
    f1 = metrics.get("f1", 0)
    recall = metrics.get("recall", 0)
    
    if f1 >= 0.80 and recall >= 0.80:
        return "OK"
    elif f1 >= 0.65 or recall >= 0.50:
        return "WARNING"
    else:
        return "CRITICAL"

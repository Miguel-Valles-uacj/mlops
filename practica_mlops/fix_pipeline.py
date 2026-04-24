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
    return "Equipo 1"


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
    # Pista: usa stats.ks_2samp(df_train[col], df_prod[col]) OK
    # Pista: las columnas numericas son: precio_unitario, descuento_pct,
    #        inventario_prev, es_temporada_alta OK
    resultado = {}
    columnas_numericas = ["precio_unitario", "descuento_pct", "inventario_prev", "es_temporada_alta"]
    for col in columnas_numericas:
        ks_stat, p_val = stats.ks_2samp(df_train[col], df_prod[col])
        resultado[col] = {
            "p_value": p_val,
            "statistic": ks_stat,
            "drift": p_val < 0.05
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

    # TODO: implementa las correcciones aqui
    # Pista 1: para detectar si el precio esta en USD, compara el rango de
    #          precio_unitario en produccion vs entrenamiento
    # Pista 2: el tipo de cambio aproximado es 17.5 pesos por dolar
    # Pista 3: para la temporada alta, revisa la columna es_temporada_alta
    #          y los valores de semana

    # Detectar si el precio esta en USD
    precio_unitario_train = df_train["precio_unitario"]
    precio_unitario_prod = df["precio_unitario"]
    if precio_unitario_prod.mean() < precio_unitario_train.mean():
        df["precio_unitario"] = df["precio_unitario"] * 17.5

    # Detectar temporada alta
    df["es_temporada_alta"] = df["semana"].apply(lambda x: 1 if x >= 45 else 0)

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

    # TODO: define tus umbrales y retorna el estado correspondiente
    # Ejemplo de estructura (cambia los valores):
    #
    # if f1 >= ???:
    #     return "OK"
    # elif f1 >= ???:
    #     return "WARNING"
    # else:
    #     return "CRITICAL"
  # Umbrales definidos basados en F1 (métrica balanceada para clasificación):
    # - OK (F1 >= 0.85): modelo con desempeño excelente, confiable en producción
    # - WARNING (F1 >= 0.75): degradación moderada, requiere monitoreo e investigación
    # - CRITICAL (F1 < 0.75): modelo no confiable, requiere reentrenamiento inmediato
    #
    # Se prioriza F1 porque balancea precision y recall, evitando sesgos en datos
    # desbalanceados. También se verifica accuracy como métrica complementaria.
    
    if f1 >= 0.85 and accuracy >= 0.80:
        return "OK"
    elif f1 >= 0.75 and accuracy >= 0.70:
        return "WARNING"
    else:
        return "CRITICAL"
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
    return "Equipo 1 — Luis Flores, Evelyn Bravo, Rogelio Saucedo"


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
    resultado = {}

    columnas_numericas = ["precio_unitario", "descuento_pct",
                          "inventario_prev", "es_temporada_alta"]

    for col in columnas_numericas:
        if col not in df_train.columns or col not in df_prod.columns:
            continue

        stat, p_value = stats.ks_2samp(
            df_train[col].dropna().values,
            df_prod[col].dropna().values
        )

        resultado[col] = {
            "p_value":   round(float(p_value), 4),
            "statistic": round(float(stat), 4),
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

    # ── Falla 2: precio_unitario en USD en vez de MXN ────────────────────────
    # No todos los registros tienen el precio en USD, solo algunos.
    # Lo detectamos comparando contra el minimo del dataset de entrenamiento:
    # si un precio esta por debajo del minimo posible en MXN (segun train),
    # significa que llego en dolares y hay que convertirlo.
    # Pista del enunciado: "compara el rango de precio_unitario en produccion
    # vs entrenamiento" — aqui comparamos el minimo del rango.
    min_precio_train = df_train["precio_unitario"].min()

    if min_precio_train <= 0:
        raise ValueError(
            "El minimo de precio_unitario en entrenamiento es cero o negativo. "
            "Los datos de entrenamiento son invalidos."
        )

    mask_usd = df["precio_unitario"] < min_precio_train
    df.loc[mask_usd, "precio_unitario"] = df.loc[mask_usd, "precio_unitario"] * 17.5

    # ── Falla 1: drift estacional ─────────────────────────────────────────────
    # El diagnostico confirma que es_temporada_alta ya esta en 1 para semanas
    # 45-52 en produccion — esos valores son correctos y no necesitan cambio.
    # La raiz del drift es que el modelo fue entrenado con es_temporada_alta=0
    # en todos sus datos (semanas 1-40 no tienen temporada alta), por lo que
    # no aprendio patrones de esa variable. El fix real es corregir el precio
    # para que el modelo pueda clasificar correctamente usando las otras features.

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

    Justificacion:
        El modelo base tenia Accuracy ~0.91 y F1 ~0.86 en semanas sin fallas
        (semanas 38-44 de la curva de degradacion).
        - OK       (F1 >= 0.75 y Accuracy >= 0.80): funcionando bien
        - WARNING  (F1 >= 0.50 y Accuracy >= 0.65): degradacion notable
        - CRITICAL (debajo de eso): falla grave, no usar para decisiones
    """
    accuracy  = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall    = metrics.get("recall", 0)
    f1        = metrics.get("f1", 0)

    if f1 >= 0.75 and accuracy >= 0.80:
        return "OK"
    elif f1 >= 0.50 and accuracy >= 0.65:
        return "WARNING"
    else:
        return "CRITICAL"
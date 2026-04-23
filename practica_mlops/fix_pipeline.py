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
    return "Equipo 8 -- Gabriel, Ian, Angel"


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
    if df_train.empty or df_prod.empty:
        raise ValueError(
            "detect_drift: uno o ambos DataFrames están vacíos. "
            "Verifica que los archivos de datos se cargaron correctamente."
        )
    
    columnas_numericas = ["precio_unitario","descuento_pct","inventario_prev","es_temporada_alta"]
    
    for col in columnas_numericas:
        if col not in df_train.columns:
            raise ValueError(
                f"detect_drift: columna '{col}' no encontrada en df_train. "
                f"Columnas disponibles: {list(df_train.columns)}"
            )
        if col not in df_prod.columns:
            raise ValueError(
                f"detect_drift: columna '{col}' no encontrada en df_prod. "
                f"Columnas disponibles: {list(df_prod.columns)}"
            )
    
    resultado = {}

    
    for col in columnas_numericas:
        statistic, p_value= stats.ks_2samp(df_train[col],df_prod[col])
        resultado[col] = {
            "p_value": round(float(p_value),6),
            "statistic": round(float(statistic),4),
            "drift": p_value < 0.05
        }
    
    # raise NotImplementedError("Implementa detect_drift()")

    return resultado


# =============================================================================
# FUNCION 2 — Correccion de datos
# =============================================================================

def _detectar_precio_en_usd(df_prod:pd.DataFrame, df_train:pd.DataFrame)-> bool:
    """
    Compara la mediana de los datos de precio en producción vs entrenamiento. Si la mediana de los precios de producción es menos que la mitas
    del entramiento, se asume que los precios vienen en USD
    """
    
    mediana_train = df_train["precio_unitario"].median()
    mediana_prod = df_prod["precio_unitario"].median()
    
    return mediana_prod < (mediana_train/2)

def _corregir_precio_usd(df: pd.DataFrame, df_train: pd.DataFrame, tipo_cambio:float = 17.5) -> pd.DataFrame:
    """
    Filas cuyo valor este bajo del minimo, se multiplican
    """
    precio_minimo_historico = df_train["precio_unitario"].min()
    filas_en_usd = df["precio_unitario"] < precio_minimo_historico
    n_filas = filas_en_usd.sum()
    
    if n_filas==0:
        return df # No hay nada que corregir
    
    df.loc[filas_en_usd, "precio_unitario"] = (
        df.loc[filas_en_usd, "precio_unitario"] * tipo_cambio
    )
    return df

def _corregir_temporada_alta(df: pd.DataFrame) -> pd.DataFrame:
    df["es_temporada_alta"] = 0
    return df



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
   

    # TODO: implementa las correcciones aqui
    # Pista 1: para detectar si el precio esta en USD, compara el rango de
    #          precio_unitario en produccion vs entrenamiento
    # Pista 2: el tipo de cambio aproximado es 17.5 pesos por dolar
    # Pista 3: para la temporada alta, revisa la columna es_temporada_alta
    #          y los valores de semana

    if df_prod is None or df_train is None:
        raise ValueError(
            "fix_data: df_prod y df_train no pueden ser None."
        )
    if df_prod.empty:
        raise ValueError(
            "fix_data: df_prod está vacío. No hay datos de producción que corregir."
        )
    if df_train.empty:
        raise ValueError(
            "fix_data: dt_train está vacío. Se necesita como referencia para la corrección."
        )
        
    columnas_requeridas = ["precio_unitario","es_temporada_alta"]
    for col in columnas_requeridas:
        if col not in df_prod.columns:
            raise ValueError(
                f"fix_data: columna requerida '{col}' no encontrada en df_prod. "
                f"Columnas disponibles: {list(df_prod.columns)}"
            )
            
    df = df_prod.copy()
    
    #Precio en USD
    if _detectar_precio_en_usd(df,df_train):
        df = _corregir_precio_usd(df,df_train,tipo_cambio=17.5)
        
    # es_temporal_alta no es 1 en entrenamiento
    df = _corregir_temporada_alta(df)

    # raise NotImplementedError("Implementa fix_data()")

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
    

    # TODO: define tus umbrales y retorna el estado correspondiente
    # Ejemplo de estructura (cambia los valores):
    #
    # if f1 >= ???:
    #     return "OK"
    # elif f1 >= ???:
    #     return "WARNING"
    # else:
    #     return "CRITICAL"
    
    if not isinstance(metrics,dict):
        raise ValueError(
            "check_model_health: 'metrics' debe ser un diccionario con keys "
            "'accuracy','precision', 'recall', 'f1'."
        )
        
    for key in ["accuracy","precision","recall","f1"]:
        if key not in metrics:
            raise ValueError(
                f"check_model_health: falta la métrica '{key}' en el diccionario. "
                f"Keys recibidas: {list(metrics.keys())}"
            )
        valor = metrics[key]
        if not isinstance(valor, (int,float)) or not (0.0 <= valor <= 1.0):
            raise ValueError(
                f"check_model_health: el valor de '{key}' debe ser un número "
                f"entre 0 y 1. Valor recibido: {valor}"
            )
    """ accuracy  = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall    = metrics.get("recall", 0)
    f1        = metrics.get("f1", 0) """
    f1 = metrics["f1"]
    accuracy = metrics["accuracy"]
    
    if f1>=0.80 and accuracy >= 0.85:
        return "OK"
    elif f1 >= 0.60 and accuracy >= 0.70:
        return "WARNING"
    else:
        return "CRITICAL"

    #raise NotImplementedError("Implementa check_model_health() con tus umbrales justificados")

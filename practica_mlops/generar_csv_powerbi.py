"""
generar_csv_powerbi.py
Genera el archivo reports/metricas_powerbi.csv para Power BI.
Ejecuta: python generar_csv_powerbi.py
"""

import json
import pandas as pd
import os

REPORTE_PATH = "reports/reporte_equipo.json"
OUTPUT_PATH  = "reports/metricas_powerbi.csv"

# ── Cargar reporte ────────────────────────────────────────────────────────────
if not os.path.exists(REPORTE_PATH):
    print("[ERROR] No se encontro reports/reporte_equipo.json")
    print("Asegurate de haber corrido: python run_pipeline.py --fixed")
    exit(1)

with open(REPORTE_PATH, "r", encoding="utf-8") as f:
    reporte = json.load(f)

equipo    = reporte["equipo"]
base      = reporte["metricas_baseline"]
fix       = reporte["metricas_con_fix"]
semanas   = reporte["semanas_evaluadas"]

# ── Tabla 1: curva de degradacion por semana ─────────────────────────────────
# Necesitamos las metricas por semana — las reconstruimos desde los datos
import sys
sys.path.insert(0, ".")

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import fix_pipeline as fp

FEATURES = ["precio_unitario", "descuento_pct", "inventario_prev", "es_temporada_alta"]

df_train = pd.read_csv("data/ventas_train.csv")
df_prod  = pd.read_csv("data/ventas_produccion.csv")
df_fix   = fp.fix_data(df_prod.copy(), df_train.copy())
model    = joblib.load("models/modelo_base.pkl")

filas_curva = []
for sem in semanas:
    # baseline
    mask = df_prod["semana"] == sem
    y    = df_prod[mask]["ventas_categoria"]
    yp   = model.predict(df_prod[mask][FEATURES])
    acc_b = round(accuracy_score(y, yp), 4)
    f1_b  = round(f1_score(y, yp, pos_label="ALTA", zero_division=0), 4)

    # con fix
    mask_f = df_fix["semana"] == sem
    y_f    = df_fix[mask_f]["ventas_categoria"]
    yp_f   = model.predict(df_fix[mask_f][FEATURES])
    acc_f  = round(accuracy_score(y_f, yp_f), 4)
    f1_f   = round(f1_score(y_f, yp_f, pos_label="ALTA", zero_division=0), 4)

    filas_curva.append({
        "equipo":       equipo,
        "semana":       sem,
        "accuracy_base": acc_b,
        "f1_base":       f1_b,
        "accuracy_fix":  acc_f,
        "f1_fix":        f1_f,
    })

df_curva = pd.DataFrame(filas_curva)

# ── Tabla 2: resumen de metricas antes vs despues ────────────────────────────
filas_resumen = []
for metrica in ["accuracy", "precision", "recall", "f1"]:
    filas_resumen.append({
        "equipo":   equipo,
        "metrica":  metrica.capitalize(),
        "baseline": base[metrica],
        "con_fix":  fix[metrica],
        "mejora":   round(fix[metrica] - base[metrica], 4),
    })

df_resumen = pd.DataFrame(filas_resumen)

# ── Guardar ───────────────────────────────────────────────────────────────────
os.makedirs("reports", exist_ok=True)

# Power BI acepta un solo CSV con una columna que identifique la tabla
# Guardamos dos archivos separados para mayor claridad
df_curva.to_csv("reports/curva_powerbi.csv",   index=False, encoding="utf-8")
df_resumen.to_csv("reports/resumen_powerbi.csv", index=False, encoding="utf-8")

print(f"Archivos generados:")
print(f"  reports/curva_powerbi.csv   ({len(df_curva)} filas)")
print(f"  reports/resumen_powerbi.csv ({len(df_resumen)} filas)")
print(f"\nEquipo: {equipo}")
print(f"Semanas evaluadas: {semanas[0]} - {semanas[-1]}")
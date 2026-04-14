"""
SalesPredict — Pipeline de evaluacion
======================================
Uso:
    python run_pipeline.py           # evalua modelo base en produccion
    python run_pipeline.py --fixed   # evalua con el fix del equipo aplicado
"""

import sys
import json
import argparse
import os
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ── Rutas ─────────────────────────────────────────────────────────────────────
DATA_TRAIN  = "data/ventas_train.csv"
DATA_PROD   = "data/ventas_produccion.csv"
MODEL_PATH  = "models/modelo_base.pkl"
REPORTS_DIR = "reports"
FEATURES    = ["precio_unitario", "descuento_pct", "inventario_prev", "es_temporada_alta"]

os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Argumentos ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--fixed", action="store_true", help="Aplica fix_pipeline.py antes de evaluar")
args = parser.parse_args()

# ── Carga de datos y modelo ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SalesPredict — Reporte de Metricas")
print("=" * 60)

df_train = pd.read_csv(DATA_TRAIN)
df_prod  = pd.read_csv(DATA_PROD)
model    = joblib.load(MODEL_PATH)

print(f"\n  Datos de entrenamiento : {len(df_train):,} registros (semanas 1-40)")
print(f"  Datos de produccion    : {len(df_prod):,} registros (semanas 38-52)")

# ── Aplicar fix si corresponde ────────────────────────────────────────────────
fix_applied = False
team_name   = "Sin nombre"

if args.fixed:
    try:
        import fix_pipeline as fp
        team_name   = fp.get_team_name()
        df_prod_fix = fp.fix_data(df_prod.copy(), df_train.copy())
        drift_report = fp.detect_drift(df_train[FEATURES], df_prod_fix[FEATURES])
        health_before = None
        fix_applied   = True
        print(f"\n  Fix aplicado por: {team_name}")
        print("\n  Drift detectado post-fix:")
        for col, resultado in drift_report.items():
            estado = "DRIFT" if resultado["drift"] else "OK"
            print(f"    {col:<22} p-value={resultado['p_value']:.4f}  [{estado}]")
    except ImportError:
        print("\n  [ERROR] No se encontro fix_pipeline.py")
        print("  Asegurate de estar en el directorio raiz del proyecto.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  [ERROR] en fix_pipeline.py: {e}")
        sys.exit(1)
else:
    df_prod_fix = df_prod.copy()
    print("\n  Modo: baseline (sin fix)")

# ── Metricas globales ─────────────────────────────────────────────────────────
def calcular_metricas(df, label=""):
    X = df[FEATURES]
    y = df["ventas_categoria"]
    y_pred = model.predict(X)
    return {
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, pos_label="ALTA", zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, pos_label="ALTA", zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred, pos_label="ALTA", zero_division=0), 4),
    }

metricas_base = calcular_metricas(df_prod)
metricas_fix  = calcular_metricas(df_prod_fix) if fix_applied else None

print("\n" + "-" * 60)
print(f"  {'Metrica':<12} {'Baseline':>10}", end="")
if fix_applied:
    print(f"  {'Con Fix':>10}  {'Mejora':>8}", end="")
print()
print("-" * 60)

for key in ["accuracy", "precision", "recall", "f1"]:
    base_val = metricas_base[key]
    row = f"  {key.capitalize():<12} {base_val:>10.4f}"
    if fix_applied:
        fix_val  = metricas_fix[key]
        mejora   = fix_val - base_val
        signo    = "+" if mejora >= 0 else ""
        row     += f"  {fix_val:>10.4f}  {signo}{mejora:>7.4f}"
    print(row)

print("-" * 60)

# ── Check de salud ────────────────────────────────────────────────────────────
if fix_applied:
    try:
        metricas_para_health = metricas_fix
        estado = fp.check_model_health(metricas_para_health)
        print(f"\n  Estado del modelo (post-fix): [{estado}]")
    except Exception as e:
        print(f"\n  [AVISO] check_model_health() fallo: {e}")

# ── Curva de degradacion semana a semana ──────────────────────────────────────
semanas    = sorted(df_prod["semana"].unique())
acc_curve  = []
f1_curve   = []
acc_fix_curve = []
f1_fix_curve  = []

for sem in semanas:
    mask = df_prod["semana"] == sem
    m    = calcular_metricas(df_prod[mask])
    acc_curve.append(m["accuracy"])
    f1_curve.append(m["f1"])

    if fix_applied:
        mask_fix = df_prod_fix["semana"] == sem
        mf       = calcular_metricas(df_prod_fix[mask_fix])
        acc_fix_curve.append(mf["accuracy"])
        f1_fix_curve.append(mf["f1"])

# ── Grafica ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("SalesPredict — Curva de Degradacion por Semana", fontsize=13, fontweight="bold", y=1.01)

colores = {"base_acc": "#E8593C", "base_f1": "#1B2A4A",
           "fix_acc":  "#2E7A4A", "fix_f1":  "#C47A2B"}

for ax, metric_base, metric_fix, titulo, ylabel in [
    (axes[0], acc_curve, acc_fix_curve if fix_applied else None, "Accuracy", "Accuracy"),
    (axes[1], f1_curve,  f1_fix_curve  if fix_applied else None, "F1 Score", "F1"),
]:
    color_base = colores["base_acc"] if ylabel == "Accuracy" else colores["base_f1"]
    ax.plot(semanas, metric_base, "o-", color=color_base, linewidth=2,
            markersize=5, label="Baseline", zorder=3)

    if metric_fix is not None:
        color_fix = colores["fix_acc"] if ylabel == "Accuracy" else colores["fix_f1"]
        ax.plot(semanas, metric_fix, "s--", color=color_fix, linewidth=2,
                markersize=5, label="Con fix", zorder=3)

    # Linea vertical donde empiezan las fallas
    ax.axvline(x=44.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(44.7, 0.05, "Fallas\ninyectadas", fontsize=8, color="gray", va="bottom")

    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_xlabel("Semana", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(semanas)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
curva_path = os.path.join(REPORTS_DIR, "curva_degradacion.png")
plt.savefig(curva_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Curva guardada en: {curva_path}")

# ── Reporte JSON (solo con --fixed) ──────────────────────────────────────────
if fix_applied:
    reporte = {
        "equipo": team_name,
        "timestamp": datetime.now().isoformat(),
        "metricas_baseline": metricas_base,
        "metricas_con_fix": metricas_fix,
        "mejora": {k: round(metricas_fix[k] - metricas_base[k], 4) for k in metricas_base},
        "semanas_evaluadas": semanas,
        "drift_detectado": drift_report,
    }

    reporte_path = os.path.join(REPORTS_DIR, "reporte_equipo.json")

    def serializable(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        raise TypeError(f"No serializable: {type(obj)}")

    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False, default=serializable)
    print(f"  Reporte JSON guardado en: {reporte_path}")

print("\n" + "=" * 60 + "\n")

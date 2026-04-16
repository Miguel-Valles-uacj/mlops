# mlops

# SalesPredict — Detección de Drift en Ventas

> **Práctica de Clase — MLOps**  
> Universidad Autónoma de Ciudad Juárez · Métricas del Software · 2026

---

## Descripción

Este repositorio contiene la práctica de MLOps basada en **SalesPredict**, un modelo de clasificación que predice si una semana de ventas en una cadena retail será **ALTA** o **BAJA**.

El escenario simula un equipo MLOps real que debe **diagnosticar, documentar y reparar un pipeline en producción** tras una caída crítica de rendimiento:

| Métrica | Antes del drift | Después del drift |
|---------|----------------|-------------------|
| Accuracy | ~0.91 | ~0.63 |
| F1 Score | ~0.89 | ~0.51 |

---

## Estructura del Repositorio

```
mlops/practica_mlops/
 models/
    modelo_base.pkl          # Modelo Decision Tree entrenado
 reports/
    curva_degradacion.png    # Curva de degradación del modelo
    metricas_powerbi.csv     # Datos listos para Power BI
 run_pipeline.py              # Pipeline principal (no modificar)
 fix_pipeline.py              #  Archivo a editar por cada equipo
 reporte_equipo.json          # Generado automáticamente al correr --fixed
 reporte_equipo.pbix          # Reporte Power BI del equipo
 requirements.txt
```

---

## Setup del Entorno

> Se recomienda guardar la carpeta en `C:/Users/<usuario>/`

### 1. Clonar el repositorio

```bash
git clone https://github.com/Miguel-Valles-uacj/mlops
cd mlops/practica_mlops
git checkout equipo-N     # Reemplaza N con tu número de equipo
git branch                # Debe aparecer: * equipo-N
```

### 2. Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

### 3. Correr el pipeline base

```bash
python run_pipeline.py
# Métricas en consola + curva en reports/curva_degradacion.png
```

### 4. Correr el pipeline con tu fix

```bash
python run_pipeline.py --fixed
# Genera reporte_equipo.json con métricas antes/después
```

### 5. Subir el trabajo

```bash
git add .
git commit -m "feat: fix pipeline equipo-N"
git push origin equipo-N
```

---

## Dataset — Campos del Sistema

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `semana` | INTEGER | Número de semana del año |
| `tienda_id` | INTEGER | Identificador de sucursal |
| `categoria` | STRING | Categoría del producto |
| `precio_unitario` | FLOAT | Precio por unidad (puede estar en USD en producción) |
| `descuento_pct` | FLOAT | Porcentaje de descuento aplicado |
| `inventario_prev` | INTEGER | Inventario de la semana anterior |
| `es_temporada_alta` | BOOLEAN | 1 = temporada alta (Nov-Dic), 0 = resto del año |
| `ventas_categoria` | STRING | **Variable objetivo**: ALTA o BAJA |

**Modelo base:** Decision Tree (`scikit-learn`) — `models/modelo_base.pkl`

---

## Flujo de Trabajo — Las 4 Fases

```
F1 → Detectar el problema
F2 → Proponer una solución  (sin IA)
F3 → Implementar la solución
F4 → Evaluar y reflexionar
```

### F1 — Detectar el Problema
Corre `run_pipeline.py`, analiza las métricas y responde las preguntas guía del documento sobre métricas, datos y pipeline.

### F2 — Proponer una Solución
Sin ayuda de IA. Escribe tu **hipótesis de causa raíz** y tu **plan de fix** antes de abrir `fix_pipeline.py`.

### F3 — Implementar la Solución
Edita `fix_pipeline.py` completando las funciones vacías:
- `detect_drift(df_train, df_prod)` — usando el **test KS de scipy**
- Función de fix según tu hipótesis de F2
- Funciones encadenadas, no script lineal
- Excepciones descriptivas para datos inválidos (no `try/except` vacíos)

Librerías permitidas: únicamente las de `requirements.txt` (`pandas`, `scipy`, `scikit-learn`).

### F4 — Evaluar y Reflexionar
Corre `run_pipeline.py --fixed` y analiza los resultados. Define umbrales para `check_model_health()` (OK / WARNING / CRITICAL) y toma una decisión de negocio fundamentada en los números.

---

## Entregables por Equipo

| Archivo | Descripción |
|---------|-------------|
| `fix_pipeline.py` | Solución implementada con funciones estructuradas |
| `reporte_equipo.json` | Generado automáticamente con métricas antes/después |
| `reporte_equipo.pbix` | Reporte Power BI con visualizaciones |
| Push a `equipo-N` | Rama con ambos archivos entregados |

---

## Visualización en Power BI

Al terminar la Fase 4, importa `reports/metricas_powerbi.csv` en **Power BI Desktop** y crea:

1. **Gráfica de líneas** — Curva de degradación (`semana` vs `accuracy_base` + `accuracy_fix`)
2. **Gráfica de barras agrupadas** — Métricas antes vs después (Accuracy, Precision, Recall, F1)
3. **Tarjeta (Card)** — Nombre del equipo

Guarda como `reporte_equipo.pbix` dentro de la carpeta del proyecto.

>  Power BI Desktop solo está disponible para Windows.

---

## Reglas Generales

-  Solo modificar `fix_pipeline.py`
-  No modificar `run_pipeline.py` ni los archivos de datos
-  En F1 y F2 **no está permitido el uso de IA generativa** (ChatGPT, Copilot, Claude, Gemini, etc.)
-  En F3 se puede consultar documentación oficial de `scikit-learn`, `pandas` y `scipy`
-  Hacer commit al terminar cada función antes de pasar a la siguiente fase
-  Lanzar excepciones descriptivas si los datos son inválidos — nunca silenciar errores

---

## Equipos

Cada equipo trabaja en su propia rama: `equipo-1`, `equipo-2`, ..., `equipo-N`.

1 MARCOS
2 ALAN ALEJANDRO
3 VILLEDO
4 EDWIN
5 JESUS ANDRE
6 LUIS UBALDO
7 ERICK
8 GABRIEL
9 JENNIFER

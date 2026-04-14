# SalesPredict — Práctica MLOps

> Universidad Autónoma de Ciudad Juárez · Métricas del Software · 2026

## Setup rápido

```bash
# 1. Clonar y entrar a tu rama
git clone https://github.com/[equipo-lider]/salespredict.git
cd salespredict
git checkout equipo-N        # reemplaza N con tu número
git branch                   # verifica: * equipo-N

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Correr el pipeline base (Fase 1)
python run_pipeline.py

# 4. Correr con tu fix aplicado (Fase 4)
python run_pipeline.py --fixed
```

## Estructura del proyecto

```
salespredict/
├── run_pipeline.py          # NO modificar
├── fix_pipeline.py          # Tu trabajo va aquí
├── data/
│   ├── ventas_train.csv     # Datos de entrenamiento (semanas 1-40)
│   └── ventas_produccion.csv# Datos de producción con fallas (semanas 38-52)
├── models/
│   └── modelo_base.pkl      # Decision Tree pre-entrenado
├── reports/                 # Se genera al correr el pipeline
│   ├── curva_degradacion.png
│   └── reporte_equipo.json
└── requirements.txt
```

## Flujo de trabajo

| Fase | Qué hacen | Comando |
|------|-----------|---------|
| F1   | Detectar el problema | `python run_pipeline.py` |
| F2   | Proponer solución | *(sin código)* |
| F3   | Implementar fix | Editar `fix_pipeline.py` |
| F4   | Evaluar mejora | `python run_pipeline.py --fixed` |

## Subir su trabajo

```bash
git add .
git commit -m "feat: fix pipeline equipo-N"
git push origin equipo-N
```

## Reglas

- Trabajar **únicamente** en `fix_pipeline.py`
- En F1 y F2: sin IA generativa
- En F3: solo documentación oficial de scikit-learn, pandas y scipy
- Las respuestas a las preguntas van en el documento de la práctica

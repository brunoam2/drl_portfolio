# Optimización de carteras con aprendizaje por refuerzo profundo (DRL)

Este repositorio implementa un sistema de gestión de carteras mediante aprendizaje por refuerzo profundo. Utiliza un entorno compatible con Gym y se apoya en el algoritmo SAC para asignar dinámicamente pesos entre SPY, TLT, GLD y efectivo.

## Cómo ejecutar el proyecto

Los principales pasos de trabajo se encapsulan en scripts dentro de la carpeta `scripts/`:

### `01_download_data.py`
Descarga los precios históricos de los activos y los indicadores macroeconómicos necesarios.

```bash
python scripts/01_download_data.py
```

### `02_explore_data.py`
Genera las visualizaciones utilizadas en la memoria del proyecto a partir del archivo `datasets/combined_data.csv`.

```bash
python scripts/02_explore_data.py
```

### `03_test_env.py`
Crea el entorno de inversión y ejecuta dos políticas baseline predefinidas (pesos iguales y SPY únicamente) para comprobar que el entorno funciona correctamente.

```bash
python scripts/03_test_env.py
```

### `04_train_agent.py`
Entrena un agente con SAC. El script permite ajustar, entre otros parámetros, los rangos de entrenamiento y validación, la semilla, la frecuencia de evaluación y los pasos totales.

```bash
python scripts/04_train_agent.py
```

### `05_evaluate_agent.py`
Carga un modelo entrenado y lo evalúa frente a las estrategias clásicas implementadas en `src/benchmarks.py`.

```bash
python scripts/05_evaluate_agent.py
```

## 📁 Estructura de carpetas

```
drl_portfolio/
│
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_explore_data.py
│   ├── 03_test_env.py
│   ├── 04_train_agent.py
│   └── 05_evaluate_agent.py
│
├── src/
│   ├── data.py                # Carga y preparación de datasets
│   ├── observation_builder.py # Construcción del estado observable
│   ├── portfolio_env.py       # Entorno personalizado de inversión
│   ├── metrics.py             # Métricas de rendimiento
│   ├── benchmarks.py          # Estrategias de referencia
│   ├── train.py               # Lógica de entrenamiento y evaluación
│   └── visualizations.py      # Generación de gráficos de métricas
│
├── models/                    # Modelos entrenados
├── datasets/                  # Datos brutos y procesados
├── results/
│   ├── explore_data/          # Gráficos del script 02_explore_data
│   ├── model_training/        # Métricas y logs de entrenamiento
│   ├── model_evaluation/      # Resultados de evaluación del agente
│   └── visualizations/        # Comparativas generadas por visualizations.py
```

## Estado del proyecto

* Entorno Gym funcional y políticas baseline incluidas.
* Entrenamiento configurable mediante `train.py`.
* Evaluación y comparación con benchmarks tradicionales.
* Herramientas para generar visualizaciones de entrenamiento y evaluación.

## Lógica interna

* **`portfolio_env.py`**: define la dinámica de la cartera, las recompensas y los costes de transacción.
* **`observation_builder.py`**: crea las observaciones que recibe el agente a partir de precios, indicadores y datos macro.
* **`metrics.py`**: calcula retorno final, ratio de Sharpe, drawdown y otras métricas de rendimiento.
* **`benchmarks.py`**: implementa las estrategias clásicas contra las que se evalúa el agente.
* **`train.py`**: orquesta el entrenamiento y la validación del modelo.

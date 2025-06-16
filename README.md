# Optimización de carteras con aprendizaje por refuerzo profundo (DRL)

Este proyecto implementa un sistema de gestión de carteras financieras mediante aprendizaje por refuerzo profundo, entrenado sobre datos de mercado reales. Utiliza un entorno personalizado compatible con Gym y se basa en el algoritmo SAC para entrenar un agente que asigna dinámicamente pesos entre activos financieros (oro, renta fija, renta variable y efectivo).

> ⚠️ **Este README es provisional**. Aún faltan por completar algunos scripts (`02_explore_data.py`, `05_evaluate_agent.py`). Se actualizará con instrucciones completas en cuanto estén finalizados.

---

## Cómo ejecutar el proyecto

Todas las acciones relevantes están organizadas en scripts dentro de la carpeta `scripts/`. Cada uno de ellos representa una etapa del flujo completo de trabajo:

### `01_download_data.py`  
Descarga y guarda datos históricos de activos financieros (Yahoo Finance) y variables macroeconómicas (FRED).

```bash
python scripts/01_download_data.py
````

---

### `02_explore_data.py`

(En desarrollo) Realizará una exploración inicial de los datos: distribución de retornos, visualización de variables macro, etc.

---

### `03_test_env.py`

Permite probar el entorno de inversión con un agente aleatorio o con reglas fijas, para verificar su correcto funcionamiento.

```bash
python scripts/03_test_env.py
```

---

### `04_train_agent.py`

Entrena un agente con el algoritmo **Soft Actor-Critic (SAC)** utilizando el entorno personalizado. Contiene parámetros ajustables por el usuario como el tamaño de la ventana de observaciones, la frecuencia de rebalanceo o el coste de transacción. 

```bash
python scripts/04_train_agent.py
```

---

### `05_evaluate_agent.py`

(En desarrollo) Permitirá cargar un modelo ya entrenado, evaluar su rendimiento en un conjunto de test y compararlo con estrategias benchmark.

---

## 📁 Estructura de carpetas

```
drl_portfolio/
│
├── scripts/                   # Scripts principales (flujo de usuario)
│   ├── 01_download_data.py
│   ├── 02_explore_data.py     ← en blanco (pendiente)
│   ├── 03_test_env.py
│   ├── 04_train_agent.py
│   ├── 05_evaluate_agent.py   ← en blanco (pendiente)
│
├── src/                       # Lógica auxiliar y clases internas
│   ├── data.py                # Funciones para cargar y preparar datasets
│   ├── observation_builder.py # Construcción del estado observable
│   ├── portfolio_env.py       # Entorno personalizado de inversión (Gym)
│   ├── evaluation.py          # Métricas: retorno, Sharpe, drawdown, etc.
│
├── models/                    # Modelos entrenados (⚠️ generados en tiempo de ejecución)
├── datasets/                  # Archivos de datos descargados y procesados (⚠️ autogenerado)
├── results/
│   ├── explore_data/          # Gráficos y estadísticas exploratorias
│   └── model_training/        # Logs de entrenamiento, evolución de métricas
```

---

## Estado actual del proyecto

* Entorno Gym personalizado y funcional.
* SAC entrenado con múltiples configuraciones.
* Validación con ventanas temporales separadas.
* Scripts de exploración y evaluación en construcción.
* Análisis detallado de pesos y comparación con benchmarks tradicionales en proceso.

---

## Lógica interna (breve)

* **`portfolio_env.py`** define las reglas del entorno: observación del estado, recompensas, penalización por rebalanceo.
* **`observation_builder.py`** construye una matriz 3D de observaciones a partir de retornos, indicadores técnicos y datos macroeconómicos.
* **`evaluation.py`** implementa métricas estándar de rendimiento ajustado por riesgo.
* **`data.py`** maneja la descarga, almacenamiento y estructuración de los datos.
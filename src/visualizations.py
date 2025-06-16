import matplotlib.pyplot as plt
import pandas as pd
import os

# Ruta base de los archivos de métricas
base_path = "results/model_training"
files = {
    "reb1_tc0": "SAC_MlpPolicy_lb30_reb1_tc0_metrics.csv",
    "reb1_tc0.001": "SAC_MlpPolicy_lb30_reb1_tc0.001_metrics.csv",
    "reb1_tc0.005":  "SAC_MlpPolicy_lb30_reb1_tc0.005_metrics.csv",
    "reb1_tc0.0075":  "SAC_MlpPolicy_lb30_reb1_tc0.0075_metrics.csv",
    "reb1_tc0.01":  "SAC_MlpPolicy_lb30_reb1_tc0.01_metrics.csv"
}

# Cargar todos los CSV en un diccionario
raw_dfs = {}
for label, filename in files.items():
    path = os.path.join(base_path, filename)
    raw_dfs[label] = pd.read_csv(path, index_col=0)
    if raw_dfs[label].empty:
        raise ValueError(f"El archivo {filename} está vacío o no se ha leído correctamente.")

# Obtener lista completa de métricas (columnas) desde el primer archivo
all_metrics = raw_dfs[list(raw_dfs.keys())[0]].columns

# Crear un DataFrame por métrica con columnas para cada modelo
metric_dfs = {}
for metric in all_metrics:
    df_metric = pd.DataFrame()
    for model_name, df in raw_dfs.items():
        df_metric[model_name] = df[metric]
    df_metric.index.name = "steps"
    df_metric = df_metric[sorted(df_metric.columns)]
    metric_dfs[metric] = df_metric

# Guardar cada DataFrame como CSV en una carpeta separada
output_dir = "results/visualizations"
os.makedirs(output_dir, exist_ok=True)

for metric, df in metric_dfs.items():
    plt.figure(figsize=(10, 6))
    for model in df.columns:
        plt.plot(df.index, df[model], label=model)
    plt.title(f"Evolución de {metric.replace('_', ' ').capitalize()}")
    plt.xlabel("Pasos de entrenamiento")
    plt.ylabel(metric.replace("_", " ").capitalize())
    plt.legend(title="Modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_evolucion.png"))
    plt.close()
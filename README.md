# Trabajo Final - MLOPS y Analítica en la Nube

## Objetivo General
Desarrollar un pipeline reproducible de machine learning que permita entrenar, evaluar y registrar un modelo, completamente automatizado mediante CI/CD usando GitHub Actions.

## Estructura del Proyecto

```
├── _init_.py
├── config.yaml
├── data.py
├── Makefile
├── model.py
├── requirements.txt
├── titanic.csv
├── src/
│   ├── train.py
│   ├── data.py
│   └── model.py
├── tests/
│   └── test_pipeline.py
└── .github/
    └── workflows/
        └── ml.yml
```

## Descripción de los Componentes

- **Dataset:** Se utiliza `titanic.csv`, un dataset de fuente libre.
- **Preprocesamiento:** Limpieza de nulos, codificación de variables categóricas y escalamiento de variables numéricas (`src/data.py`).
- **Entrenamiento:** Modelo RandomForest con hiperparámetros configurables en `config.yaml` (`src/model.py`).
- **Evaluación:** Se calculan las métricas Accuracy y F1.
- **Tracking:** MLflow registra parámetros, métricas y artefactos (modelo y scaler) en la carpeta `mlruns`.
- **Automatización:** El flujo de trabajo CI/CD está definido en `.github/workflows/ml.yml` y ejecuta instalación, pruebas, entrenamiento y guarda artefactos.

## Ejecución Local

1. Instala las dependencias:
   ```sh
   make install
   ```
2. Ejecuta el pipeline de entrenamiento:
   ```sh
   make train
   ```
3. Ejecuta las pruebas:
   ```sh
   make test
   ```
4. Revisa los experimentos y artefactos en la carpeta `mlruns`.

## Ejecución en CI/CD (GitHub Actions)

- Cada push o pull request en la rama `main` ejecuta el workflow de CI/CD.
- El workflow instala dependencias, ejecuta pruebas, limpia y prepara las carpetas de artefactos, entrena el modelo y sube los artefactos generados.
- Puedes ver los runs y artefactos en la pestaña **Actions** del repositorio.

## Configuración

- **Hiperparámetros y rutas:** Se configuran en `config.yaml`.
- **Tracking MLflow:** Local, usando `file:./mlruns`.

## Requisitos
- Dataset de libre acceso (no sklearn.datasets)
- MLflow con tracking local
- Proyecto funcional desde consola
- CI/CD activo en GitHub
- Evidencia de modelo registrado (captura de pantalla o link al run)

## Entregables
- URL del repositorio público en GitHub
- Evidencia del modelo registrado con MLflow
- Archivo .zip del proyecto


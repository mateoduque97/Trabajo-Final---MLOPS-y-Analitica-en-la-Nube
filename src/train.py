import os
import yaml
import mlflow
import mlflow.sklearn
import joblib
from data import load_data, preprocess, split
from model import build_model, evaluate

#flujo

def main(cfg_path="config.yaml"):
    print("Current working directory:", os.getcwd())
    print("Script directory:", os.path.dirname(__file__))
    # Crear carpeta artifacts absoluta relativa al archivo actual
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    print("artifacts_dir:", artifacts_dir)
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir, exist_ok=True)
    # Cargar configuración
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Configurar MLflow
    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])

    # Cargar y preparar datos
    df = load_data(cfg['dataset']['path'])
    X, y, scaler, feature_names = preprocess(df, cfg['dataset']['target'])
    X_train, X_test, y_train, y_test = split(
        X, y, cfg['train']['test_size'], cfg['train']['random_state']
    )

    # Construir modelo
    model = build_model(cfg['train'])

    # Entrenar y loguear
    with mlflow.start_run():
        # Registrar parámetros
        mlflow.log_param("model_type", cfg['train']['model']['type'])
        for k, v in cfg['train']['model'].items():
            mlflow.log_param(k, v)

        # Entrenar modelo
        model.fit(X_train, y_train)

        # Registrar métricas
        metrics = evaluate(model, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Guardar scaler como artifact
        scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        print("scaler_path:", scaler_path)
        mlflow.log_artifact(scaler_path)

        # Definir signature y input_example antes de guardar el modelo
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[:5]

        # Guardar modelo localmente en una carpeta con MLflow
        local_model_dir = os.path.join(artifacts_dir, "rf_titanic_model")
        mlflow.sklearn.save_model(
            model,
            path=local_model_dir,
            signature=signature,
            input_example=input_example
        )
        print("local_model_dir:", local_model_dir)
        # Subir la carpeta completa como artefacto
        mlflow.log_artifact(local_model_dir)

        # (Opcional) Registrar modelo en MLflow tracking server como antes
        mlflow.sklearn.log_model(
            model,
            artifact_path="rf_titanic_model",
            signature=signature,
            input_example=input_example
        )
        print("Ruta local del modelo en MLflow:", mlflow.get_artifact_uri("rf_titanic_model"))
        print("Modelo registrado en MLflow.")

        print("Run metrics:", metrics)

if __name__ == "__main__":
    main()

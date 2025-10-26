import os
import yaml
import mlflow
import mlflow.sklearn
import joblib
from data import load_data, preprocess, split
from model import build_model, evaluate

def main(cfg_path="config.yaml"):
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

    # Crear carpeta artifacts absoluta relativa al archivo actual
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir, exist_ok=True)

    # Entrenar y loguear
    with mlflow.start_run():
        # log params
        mlflow.log_param("model_type", cfg['train']['model']['type'])
        for k, v in cfg['train']['model'].items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)

        # evaluar
        metrics = evaluate(model, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # guardar scaler localmente y como artifact
        scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        # guardar modelo en artifacts y loguear
        model_path = os.path.join(artifacts_dir, cfg['output']['model_name'])
        mlflow.sklearn.save_model(model, model_path)
        mlflow.log_artifact(model_path)

        print("Run metrics:", metrics)

if __name__ == "__main__":
    main()

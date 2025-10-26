import os
import yaml
import mlflow
import mlflow.sklearn
import joblib
from data import load_data, preprocess, split
from model import build_model, evaluate

def main(cfg_path="config.yaml"):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])

    df = load_data(cfg['dataset']['path'])
    X, y, scaler, feature_names = preprocess(df, cfg['dataset']['target'])
    X_train, X_test, y_train, y_test = split(X, y, cfg['train']['test_size'], cfg['train']['random_state'])

    model = build_model(cfg['train'])

    with mlflow.start_run():
        # log params
        mlflow.log_param("model_type", cfg['train']['model']['type'])
        for k, v in cfg['train']['model'].items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # guardar scaler y model locally y como artefacto
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(scaler, "artifacts/scaler.joblib")
        mlflow.log_artifact("artifacts/scaler.joblib")

        # log model con mlflow
        mlflow.sklearn.log_model(model, cfg['output']['model_name'])

        print("Run metrics:", metrics)

if __name__ == "__main__":
    main()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def build_model(cfg):
    if cfg['model']['type'] == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=cfg['model'].get('n_estimators',100),
            max_depth=cfg['model'].get('max_depth', None),
            random_state=cfg.get('random_state', 42)
        )
    raise ValueError("Modelo no soportado")

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    return {"accuracy": float(acc), "f1": float(f1)}

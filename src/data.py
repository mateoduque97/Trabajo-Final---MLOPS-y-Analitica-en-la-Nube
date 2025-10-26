import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df, target):
    df = df.copy()
    df = df.dropna(subset=[target])  # asegurar target presente

    # Ejemplo sencillo: rellenar nulos numéricos con la mediana
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # Categóricas: rellenar con 'missing'
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna('missing')

    # Separar X/y
    X = df.drop(columns=[target])
    y = df[target]

    # Encoding simple
    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, scaler, X.columns.tolist()

def split(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

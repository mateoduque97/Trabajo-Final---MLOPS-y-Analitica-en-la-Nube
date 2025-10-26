from src.data import load_data, preprocess
import pandas as pd
import tempfile
import os

def test_small_csv(tmp_path):
    csv = tmp_path / "mini.csv"
    csv.write_text("A,B,target\n1,cat,1\n2,dog,0\n")
    df = load_data(str(csv))
    X, y, scaler, cols = preprocess(df, 'target')
    assert X.shape[0] == 2
    assert len(y) == 2

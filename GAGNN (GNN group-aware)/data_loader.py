import pandas as pd
import numpy as np

def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df

def preprocess_data(df):
    df = df.copy()
    # Log-scale amounts
    for col in ['Amount_Received', 'Amount_Paid']:
        if col in df.columns:
            df[col] = np.log1p(df[col].astype(float))
    # Dummies for categories
    cats = [c for c in ['Payment_Format', 'Receiving_Currency', 'Payment_Currency'] if c in df.columns]
    df = pd.get_dummies(df, columns=cats, dtype=float)
    return df.fillna(0)

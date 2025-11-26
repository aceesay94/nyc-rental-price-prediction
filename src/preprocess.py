import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Basic cleaning
    df = df.drop(columns=["id", "name", "host_id", "host_name"])
    df = df[df["price"] > 0]

    # Remove crazy outliers (top 1 percent)
    df = df[df["price"] < df["price"].quantile(0.99)]

    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df = df.dropna()

    return df

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

from preprocess import load_and_clean_data, build_preprocessor

def train_model(csv_path, model_output_path):
    df = load_and_clean_data(csv_path)

    target = "price"
    features = [
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "availability_365"
    ]

    data = df[features + [target]]

    X = data[features]
    y = data[target]

    # Build preprocessing
    preprocessor, num_feats, cat_feats = build_preprocessor(X)

    # Model
    model = GradientBoostingRegressor(random_state=42)

    # Pipeline
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit
    pipe.fit(X_train, y_train)

    # Evaluate
    preds = pipe.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Model Performance:")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")

    # Save model
    os.makedirs(model_output_path.rsplit('/', 1)[0], exist_ok=True)
    joblib.dump(pipe, model_output_path)

    print(f"Model saved to: {model_output_path}")

if __name__ == "__main__":
    train_model("../data/raw/AB_NYC_2019.csv", "../models/airbnb_price_model.joblib")

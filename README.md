## NYC Airbnb Price Prediction

This project builds a machine learning model to predict Airbnb listing prices in New York City using listing features such as room type, neighborhood, latitude and longitude, availability, and review metrics. It includes data cleaning, exploratory analysis, feature engineering, model comparison, and training of a production-ready regression model.


## Project Overview

The goal of this project is to analyze the NYC Airbnb Open Data (2019) dataset and estimate rental prices based on listing characteristics. This work helps identify the most important factors influencing price and demonstrates a complete end-to-end machine learning workflow using real-world data.

## Key Features

* Exploratory data analysis

* Outlier handling and missing-value imputation

* Preprocessing of numerical and categorical features

* Regression model development and evaluation

* A modular training pipeline using Scikit-Learn

* A saved model artifact ready for reuse in applications

## Dataset

NYC Airbnb Open Data (2019)
Source: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

Features used:

* neighbourhood_group

* neighbourhood

* latitude

* longitude

* room_type

* minimum_nights

* number_of_reviews

* reviews_per_month

* availability_365

Target:

* price

## Modeling Approach
Data Preparation

* Removed extreme price outliers (top 1 percent)

* Filled missing values for review-related features

* Standardized numerical features

* One-hot encoded categorical features

Models Evaluated

* Linear Regression

* Random Forest Regressor

* Gradient Boosting Regressor

Gradient Boosting achieved the strongest performance based on RMSE, MAE, and R².

Project Structure
nyc-airbnb-price-prediction/
│
├── data/
│   └── raw/AB_NYC_2019.csv
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
│
├── src/
│   ├── preprocess.py
│   └── train_model.py
│
├── models/
│   └── airbnb_price_model.joblib
│
└── README.md

## Technologies Used
* Python 

* Pandas

* NumPy

* Scikit-Learn

* Matplotlib

* Seaborn

* Jupyter Notebooks

  Author

Aboubacarr Ceesay
Master’s in Data Science Graduate Student

GitHub: https://github.com/aceesay94

LinkedIn: https://www.linkedin.com/in/aboubacarr-ceesay94/

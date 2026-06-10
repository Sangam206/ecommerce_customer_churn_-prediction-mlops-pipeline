import pandas as pd
import numpy as np 
from database.db import engine
import os
from sklearn.model_selection import train_test_split
from database.redis_client import load_df_from_redis,save_split_data_to_redis,load_split_data_from_redis,save_encoded_data_to_redis



def data_splitting():
    df = load_df_from_redis("clean_data")

    if df is None:
        raise ValueError(
            "Failed to load 'clean_data' from Redis — connection refused or key does not exist."
        )

    mk_dir = "splitting_data"
    os.makedirs(mk_dir, exist_ok=True)

    x = df.drop(columns="Churned")
    y = df["Churned"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # x_train.to_csv(os.path.join(mk_dir, "x_train.csv"), index=False)
    # x_test.to_csv(os.path.join(mk_dir, "x_test.csv"), index=False)
    # y_train.to_csv(os.path.join(mk_dir, "y_train.csv"), index=False)
    # y_test.to_csv(os.path.join(mk_dir, "y_test.csv"), index=False)

    save_split_data_to_redis(
        "split_data",
        x_train,
        x_test,
        y_train,
        y_test
    )

    print("Data split and cached successfully.")
    

def preprocess_data():
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    x_train, x_test, y_train, y_test = load_split_data_from_redis("split_data")

    try:
        x_train.drop(columns="City", inplace=True)
        x_test.drop(columns="City", inplace=True)

        cat = x_train.select_dtypes(include="object").columns
        num = x_train.select_dtypes(include=["int64", 'float64']).columns

        ct = ColumnTransformer(transformers=[
            ("cat_cols", OneHotEncoder(sparse_output=False), cat),
            ("num_cols", StandardScaler(), num)
        ])

        x_train_array = ct.fit_transform(x_train)
        x_train = pd.DataFrame(
            x_train_array,
            columns=ct.get_feature_names_out()
        )

        x_test_array = ct.transform(x_test)
        x_test = pd.DataFrame(
            x_test_array,
            columns=ct.get_feature_names_out()
        )

        mk_dir = "after encoding"
        os.makedirs(mk_dir, exist_ok=True)

        x_train.to_csv(os.path.join(mk_dir, "x_train.csv"))
        x_test.to_csv(os.path.join(mk_dir, "x_test.csv"))

        save_encoded_data_to_redis(
            "processed_data",
            x_train,
            x_test,
            y_train,
            y_test
        )

    except Exception as e:
        raise

    except Exception as e:
        raise
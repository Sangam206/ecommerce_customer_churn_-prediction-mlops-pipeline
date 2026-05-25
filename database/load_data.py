from database.db import engine
import pandas as pd


def load_data():
    try:
        data = pd.read_csv(
            'data/raw_data/ecommerce_customer_churn_dataset.csv'
        )

        data.to_sql(
            'raw_data',
            engine,
            if_exists='replace',
            index=False
        )

        print("data loaded successfully")

    except Exception as e:
        print("Error:", e)


# load_data()

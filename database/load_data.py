from database.db import engine
import pandas as pd


def load_data():
    try:
        data = pd.read_csv(
            'data/raw_data/ecommerce_customer_churn_dataset.csv'
        )

        with engine.connect() as conn:
            data.to_sql(
                'raw_data',
                conn,
                if_exists='replace',
                index=False
            )

        print("data loaded successfully")

    except Exception as e:
        print("Error:", e)


load_data()

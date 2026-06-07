import pandas as pd
import numpy as np 
from database.db import engine
from database.redis_client import save_df_to_redis


def data_cleaning():
    with engine.connect() as conn:
        df = pd.read_sql("select * from raw_data", conn)

    # remove duplicates
    df = df.drop_duplicates()

    # handle outliers
    df.loc[(df['Age'] > 100) | (df['Age'] < 10), 'Age'] = np.nan
    df.loc[(df['Cart_Abandonment_Rate'] > 100) | (df['Cart_Abandonment_Rate'] < 0), 'Cart_Abandonment_Rate'] = np.nan
    df.loc[(df['Total_Purchases'] < 0), 'Total_Purchases'] = np.nan
    df.loc[df['Discount_Usage_Rate'] > 100, 'Discount_Usage_Rate'] = np.nan

    # fill numerical columns with median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # fill categorical columns with mode (fixed: was missing assignment)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # save to DB
    with engine.connect() as conn:
        df.to_sql('clean_data', con=conn, if_exists='replace', index=False)

    # save to Redis
    save_df_to_redis("clean_data", df)
    print("Clean data saved to Redis successfully")
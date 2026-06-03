import pandas as pd
import numpy as np 
from database.db import engine

  
def data_cleaning ():
    # remove duplicate
    # data filling
    # handeling outliars
    try:
        with engine.connect() as conn:
            df=pd.read_sql("select * from raw_data ",conn)

        df=df.drop_duplicates()
        # logger.debug("duplicates value is removed from dataset")
        
        # handeling outliars
    
        df.loc[(df['Age']>100) | (df['Age']<10),'Age']=np.nan
        
        df.loc[(df['Cart_Abandonment_Rate']>100) | (df['Cart_Abandonment_Rate']<0),'Cart_Abandonment_Rate']=np.nan
        
        
        df.loc[(df['Total_Purchases']<0),'Total_Purchases']=np.nan
        
        
        df.loc[df['Discount_Usage_Rate'] > 100, 'Discount_Usage_Rate'] = np.nan
        # logger.debug("outliars is handeled")

        # data filling 
        num_cols=df.select_dtypes(include=['float64','int64'])  
        for col in num_cols:
            df[col].fillna(df[col].median(), inplace=True)
        # logger.debug("numerical columns is filled by meadian filling")
        

        cat_cols=df.select_dtypes(include=['object'])
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0])
        # logger.debug("catogorical columns is filled by mode filling")
        
        with engine.connect() as conn:
            df.to_sql(
                'clean_data',
                con=conn,
                if_exists='replace',
                index=False
            )
    
    except Exception as e:
        # logger.error("error occured:{e}")
        print("error",e)



import pandas as pd
import numpy as np 
from database.db import engine
import os
from sklearn.model_selection import train_test_split

def data_splitting():

    df=pd.read_sql("select * from clean_data",engine)

    
    # logger.debug("data splitting start")
    mk_dir='splitting data'
    os.makedirs(mk_dir,exist_ok=True)
    x=df.drop(columns='Churned')
    y=df['Churned']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
    x_train.to_csv(os.path.join(mk_dir,"x_train.csv"),index=False)
    y_train.to_csv(os.path.join(mk_dir,"y_train.csv"),index=False)
    x_test.to_csv(os.path.join(mk_dir,"x_test.csv"),index=False)
    y_test.to_csv(os.path.join(mk_dir,"y_test.csv"),index=False)
    # logger.debug("data splitting done")
    # return x_train,x_test,y_train,y_test 
# data_splitting()

def preprocess_data():
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    # df=pd.read_sql("select * from clean_data",engine)
    # print(df.isnull().sum())
    x_train=pd.read_csv("splitting data/x_train.csv")
    x_test=pd.read_csv("splitting data/x_test.csv"
)
    try:
        x_train.drop(columns="City",inplace=True)
        x_test.drop(columns="City",inplace=True)
        cat=x_train.select_dtypes(include="object").columns
        num=x_train.select_dtypes(include=["int64",'float64']).columns
        
        ct=ColumnTransformer(transformers=[
            ("cat_cols",OneHotEncoder(sparse_output=False),cat),
            ("num_cols",StandardScaler(),num)])
        x_train_array=ct.fit_transform(x_train)
        x_train=pd.DataFrame(x_train_array,columns=ct.get_feature_names_out())

        x_test_array=ct.transform(x_test)
        x_test=pd.DataFrame(x_test_array,columns=ct.get_feature_names_out())
        
        mk_dir="after encoding"
        os.makedirs(mk_dir,exist_ok=True)

        x_train.to_csv(os.path.join(mk_dir,"x_train.csv"))
        x_test.to_csv(os.path.join(mk_dir,"x_test.csv"))
        # logger.debug("successfully encoding")

        # return x_train,x_test
    except Exception as e:
        # logger.error("error occured: {e}")
        raise

# preprocess_data()
from sqlalchemy import create_engine 
import pandas as pd
import numpy as np 


# username='sangam'
# password='root'
# host='localhost' 
# port=3380
# database='customer_churn_database'

# engine= create_engine(
#     f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
# )
# db_url="mysql+pymysql://sangam:root@localhost:3306/customer_churn_database"
engine = create_engine(
    "mysql+pymysql://sangam:root@host.docker.internal:3306/customer_churn_database"
)

# engine = create_engine(db_url)
  
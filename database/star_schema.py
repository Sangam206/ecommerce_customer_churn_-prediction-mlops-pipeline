from sqlalchemy import create_engine, text
import pandas as pd
from datetime import date
from include.logging import logger
# Database Connection
engine = create_engine(
    "mysql+pymysql://root:root@host.docker.internal:3306/customer_star_schema"
)

# Load CSV
df = pd.read_csv(r"data\raw_data\ecommerce_customer_churn_dataset.csv")


def create_star_schema():
    logger.debug("star_schema start ")
    
    with engine.begin() as conn:

        # Disable foreign key checks
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

        # Drop tables in correct order
        conn.execute(text("DROP TABLE IF EXISTS fact_churn"))
        conn.execute(text("DROP TABLE IF EXISTS dim_financial"))
        conn.execute(text("DROP TABLE IF EXISTS dim_purchase"))
        conn.execute(text("DROP TABLE IF EXISTS dim_engagement"))
        conn.execute(text("DROP TABLE IF EXISTS dim_model"))
        conn.execute(text("DROP TABLE IF EXISTS dim_date"))
        conn.execute(text("DROP TABLE IF EXISTS dim_customer"))

        # Enable foreign key checks
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

        # ---------------- DIM CUSTOMER ----------------
        conn.execute(text("""
        CREATE TABLE dim_customer (
            customer_key INT AUTO_INCREMENT PRIMARY KEY,
            gender VARCHAR(20),
            country VARCHAR(50),
            city VARCHAR(100),
            signup_quarter VARCHAR(20),
            age INT,
            membership_years INT
        )
        """))

        # ---------------- DIM DATE ----------------
        conn.execute(text("""
        CREATE TABLE dim_date (
            date_key INT AUTO_INCREMENT PRIMARY KEY,
            full_date DATE,
            year INT,
            month INT,
            day INT,
            quarter INT
        )
        """))

        # ---------------- DIM MODEL ----------------
        conn.execute(text("""
        CREATE TABLE dim_model (
            model_key INT AUTO_INCREMENT PRIMARY KEY,
            run_id VARCHAR(100),
            model_version VARCHAR(20),
            accuracy FLOAT
        )
        """))

        # ---------------- DIM ENGAGEMENT ----------------
        conn.execute(text("""
        CREATE TABLE dim_engagement (
            engagement_key INT AUTO_INCREMENT PRIMARY KEY,
            login_frequency FLOAT,
            session_duration_avg FLOAT,
            pages_per_session FLOAT,
            mobile_app_usage FLOAT,
            social_media_engagement_score FLOAT,
            email_open_rate FLOAT,
            product_reviews_written INT
        )
        """))

        # ---------------- DIM PURCHASE ----------------
        conn.execute(text("""
        CREATE TABLE dim_purchase (
            purchase_key INT AUTO_INCREMENT PRIMARY KEY,
            total_purchases INT,
            average_order_value FLOAT,
            days_since_last_purchase INT,
            discount_usage_rate FLOAT,
            returns_rate FLOAT,
            cart_abandonment_rate FLOAT,
            wishlist_items INT,
            payment_method_diversity INT
        )
        """))

        # ---------------- DIM FINANCIAL ----------------
        conn.execute(text("""
        CREATE TABLE dim_financial (
            financial_key INT AUTO_INCREMENT PRIMARY KEY,
            lifetime_value FLOAT,
            credit_balance FLOAT,
            customer_service_calls INT
        )
        """))

        # ---------------- FACT CHURN (lean) ----------------
        conn.execute(text("""
        CREATE TABLE fact_churn (
            fact_key INT AUTO_INCREMENT PRIMARY KEY,
            customer_key INT,
            date_key INT,
            model_key INT,
            engagement_key INT,
            purchase_key INT,
            financial_key INT,
            churned INT,

            FOREIGN KEY (customer_key)   REFERENCES dim_customer(customer_key),
            FOREIGN KEY (date_key)       REFERENCES dim_date(date_key),
            FOREIGN KEY (model_key)      REFERENCES dim_model(model_key),
            FOREIGN KEY (engagement_key) REFERENCES dim_engagement(engagement_key),
            FOREIGN KEY (purchase_key)   REFERENCES dim_purchase(purchase_key),
            FOREIGN KEY (financial_key)  REFERENCES dim_financial(financial_key)
        )
        """))

    logger.debug("Star Schema Created Successfully")


def insert_dim_customer(df):
    logger.debug("data inserted to dim_customer")

    customer_df = df[[
        "Gender", "Country", "City", "Signup_Quarter",
        "Age", "Membership_Years"
    ]].copy()

    customer_df.columns = [
        "gender", "country", "city", "signup_quarter",
        "age", "membership_years"
    ]

    customer_df.to_sql(
        "dim_customer",
        con=engine,
        if_exists="append",
        index=False
    )

    logger.debug("dim_customer loaded")


def insert_dim_date():

    today = date.today()

    date_df = pd.DataFrame([{
        "full_date": today,
        "year": today.year,
        "month": today.month,
        "day": today.day,
        "quarter": (today.month - 1) // 3 + 1
    }])

    date_df.to_sql(
        "dim_date",
        con=engine,
        if_exists="append",
        index=False
    )

    print("dim_date loaded")


def insert_dim_model():

    model_df = pd.DataFrame([{
        "run_id": "run_001",
        "model_version": "v1",
        "accuracy": 0.93
    }])

    model_df.to_sql(
        "dim_model",
        con=engine,
        if_exists="append",
        index=False
    )

    print("dim_model loaded")


def insert_dim_engagement(df):

    engagement_df = df[[
        "Login_Frequency", "Session_Duration_Avg", "Pages_Per_Session",
        "Mobile_App_Usage", "Social_Media_Engagement_Score",
        "Email_Open_Rate", "Product_Reviews_Written"
    ]].copy()

    engagement_df.columns = [
        "login_frequency", "session_duration_avg", "pages_per_session",
        "mobile_app_usage", "social_media_engagement_score",
        "email_open_rate", "product_reviews_written"
    ]

    engagement_df.to_sql(
        "dim_engagement",
        con=engine,
        if_exists="append",
        index=False
    )

    print("dim_engagement loaded")


def insert_dim_purchase(df):

    purchase_df = df[[
        "Total_Purchases", "Average_Order_Value", "Days_Since_Last_Purchase",
        "Discount_Usage_Rate", "Returns_Rate", "Cart_Abandonment_Rate",
        "Wishlist_Items", "Payment_Method_Diversity"
    ]].copy()

    purchase_df.columns = [
        "total_purchases", "average_order_value", "days_since_last_purchase",
        "discount_usage_rate", "returns_rate", "cart_abandonment_rate",
        "wishlist_items", "payment_method_diversity"
    ]

    purchase_df.to_sql(
        "dim_purchase",
        con=engine,
        if_exists="append",
        index=False
    )

    print("dim_purchase loaded")


def insert_dim_financial(df):

    financial_df = df[[
        "Lifetime_Value", "Credit_Balance", "Customer_Service_Calls"
    ]].copy()

    financial_df.columns = [
        "lifetime_value", "credit_balance", "customer_service_calls"
    ]

    financial_df.to_sql(
        "dim_financial",
        con=engine,
        if_exists="append",
        index=False
    )

    print("dim_financial loaded")


def insert_fact_churn(df):

    # Fetch all dimension keys in order
    customer_keys   = pd.read_sql("SELECT customer_key   FROM dim_customer   ORDER BY customer_key",   engine)
    engagement_keys = pd.read_sql("SELECT engagement_key FROM dim_engagement ORDER BY engagement_key", engine)
    purchase_keys   = pd.read_sql("SELECT purchase_key   FROM dim_purchase   ORDER BY purchase_key",   engine)
    financial_keys  = pd.read_sql("SELECT financial_key  FROM dim_financial  ORDER BY financial_key",  engine)

    fact_df = pd.DataFrame({
        "customer_key":   customer_keys["customer_key"],
        "date_key":       1,
        "model_key":      1,
        "engagement_key": engagement_keys["engagement_key"],
        "purchase_key":   purchase_keys["purchase_key"],
        "financial_key":  financial_keys["financial_key"],
        "churned":        df["Churned"].values
    })

    fact_df.to_sql(
        "fact_churn",
        con=engine,
        if_exists="append",
        index=False
    )

    print("fact_churn loaded")


def main():

    create_star_schema()

    insert_dim_customer(df)
    insert_dim_date()
    insert_dim_model()
    insert_dim_engagement(df)
    insert_dim_purchase(df)
    insert_dim_financial(df)

    insert_fact_churn(df)

    print("ETL COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
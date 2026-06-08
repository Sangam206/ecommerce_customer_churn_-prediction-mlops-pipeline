import streamlit as st
import requests

API_URL = "http://host.docker.internal:8000/predict"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")
st.title("📊 Customer Churn Prediction")
st.markdown("Fill in the customer details below to predict churn.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    country = st.selectbox("Country", ["USA", "UK", "Canada", "Australia", "India", "Germany", "France", "Japan"])
    signup_quarter = st.selectbox("Signup Quarter", ["Q1", "Q2", "Q3", "Q4"])
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    membership_years = st.number_input("Membership Years", min_value=0, value=3)
    login_frequency = st.number_input("Login Frequency", min_value=0, value=10)
    session_duration_avg = st.number_input("Session Duration Avg (mins)", min_value=0.0, value=5.0)
    pages_per_session = st.number_input("Pages Per Session", min_value=0.0, value=4.0)
    cart_abandonment_rate = st.number_input("Cart Abandonment Rate (%)", min_value=0.0, max_value=100.0, value=30.0)
    wishlist_items = st.number_input("Wishlist Items", min_value=0, value=5)
    total_purchases = st.number_input("Total Purchases", min_value=0, value=20)

with col2:
    average_order_value = st.number_input("Average Order Value ($)", min_value=0.0, value=100.0)
    days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=0, value=15)
    discount_usage_rate = st.number_input("Discount Usage Rate (%)", min_value=0.0, max_value=100.0, value=20.0)
    returns_rate = st.number_input("Returns Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
    email_open_rate = st.number_input("Email Open Rate (%)", min_value=0.0, max_value=100.0, value=40.0)
    customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=2)
    product_reviews_written = st.number_input("Product Reviews Written", min_value=0, value=3)
    social_media_engagement_score = st.number_input("Social Media Engagement Score", min_value=0.0, value=5.0)
    mobile_app_usage = st.number_input("Mobile App Usage (hrs/week)", min_value=0.0, value=3.0)
    payment_method_diversity = st.number_input("Payment Method Diversity", min_value=0, value=2)
    lifetime_value = st.number_input("Lifetime Value ($)", min_value=0.0, value=1000.0)
    credit_balance = st.number_input("Credit Balance ($)", min_value=0.0, value=500.0)

if st.button("Predict Churn", type="primary"):
    payload = {
        "Gender": gender,
        "Country": country,
        "Signup_Quarter": signup_quarter,
        "Age": age,
        "Membership_Years": membership_years,
        "Login_Frequency": login_frequency,
        "Session_Duration_Avg": session_duration_avg,
        "Pages_Per_Session": pages_per_session,
        "Cart_Abandonment_Rate": cart_abandonment_rate,
        "Wishlist_Items": wishlist_items,
        "Total_Purchases": total_purchases,
        "Average_Order_Value": average_order_value,
        "Days_Since_Last_Purchase": days_since_last_purchase,
        "Discount_Usage_Rate": discount_usage_rate,
        "Returns_Rate": returns_rate,
        "Email_Open_Rate": email_open_rate,
        "Customer_Service_Calls": customer_service_calls,
        "Product_Reviews_Written": product_reviews_written,
        "Social_Media_Engagement_Score": social_media_engagement_score,
        "Mobile_App_Usage": mobile_app_usage,
        "Payment_Method_Diversity": payment_method_diversity,
        "Lifetime_Value": lifetime_value,
        "Credit_Balance": credit_balance,
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        if "detail" in result:
            st.error(f"API Error: {result['detail']}")
        elif result["churn_prediction"] == 1:
            st.error(f"⚠️ Churn Predicted | Probability: {result['churn_probability']:.2%}")
        else:
            st.success(f"✅ No Churn | Probability: {result['churn_probability']:.2%}")

    except Exception as e:
        st.error(f"API Error: {e}")
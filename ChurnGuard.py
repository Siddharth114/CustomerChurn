import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import preprocessing
import joblib

model = joblib.load(r"/Users/siddharth/Code/Python/CustomerChurn/logistic_regression_customer_churn_classification.sav")

def main():
    st.title('ChurnGuard: Unveiling the Crystal Ball for Customer Retention')

    st.markdown('''Are you tired of losing valuable customers and revenue? Introducing our revolutionary churn prediction model powered by
      advanced machine learning algorithms. By analyzing the details of a customer, our model accurately forecasts whether a customer is likely to churn.''')
    st.markdown('''Stay one step ahead of customer attrition and take proactive measures to retain your valuable clientele.
      With our model, you can identify potential churners and implement targeted retention strategies, 
      boosting customer satisfaction and maximizing your business's bottom line. ''')
    
    st.image(Image.open('img.jpeg'))
    add_radio = st.sidebar.radio(
        'Page:',
        ("Home", "Contact")
    )

    if add_radio=='Home':
        st.header('Insert client information below')
        st.subheader("Demographic information")
        seniorcitizen = st.radio('Is the customer a senior citizen (>65 years of age)?', ('Yes', 'No'))
        dependents = st.radio('Does the customer have dependents who use the company\'s services?', ('Yes', 'No'))

        st.subheader("Payment information")
        tenure = st.slider('How many months has the customer stayed with the company until now?', min_value=0, max_value=72, value=0)
        contract = st.radio('Type of contract', ('Month-to-month', 'One year', 'Two year'))
        paperlessbilling = st.radio('Has the customer opted for paperless billing?', ('Yes', 'No'))
        PaymentMethod = st.radio('What is the customer\'s method of payment',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
        monthlycharges = st.slider('How much is the customer charged monthly?', min_value=0, max_value=150, value=0)
        totalcharges = st.slider('How much has the customer been charged totally?',min_value=0, max_value=10000, value=0)

        st.subheader("Opted-in services")
        phoneservice = st.radio('Does the customer have phone service?', ('Yes', 'No'))
        mutliplelines = st.radio("Does the customer have multiple lines?",('Yes','No','The customer has no phone service'))
        internetservice = st.radio("Does the customer have internet service?", ('DSL', 'Fiber optic', 'No'))
        onlinesecurity = st.radio("Does the customer have online security?",('Yes','No','The customer has no internet service'))
        onlinebackup = st.radio("Does the customer have online backup?",('Yes','No','The customer has no internet service'))
        techsupport = st.radio("Has the customer opted for tech support?", ('Yes','No','The customer has no internet service'))
        streamingtv = st.radio("Does the customer stream television?", ('Yes','No','The customer has no internet service'))
        streamingmovies = st.radio("Does the customer stream movies?", ('Yes','No','The customer has no internet service'))

        input_data = {
                'SeniorCitizen': seniorcitizen,
                'Dependents': dependents,
                'tenure':tenure,
                'PhoneService': phoneservice,
                'MultipleLines': mutliplelines,
                'InternetService': internetservice,
                'OnlineSecurity': onlinesecurity,
                'OnlineBackup': onlinebackup,
                'TechSupport': techsupport,
                'StreamingTV': streamingtv,
                'StreamingMovies': streamingmovies,
                'Contract': contract,
                'PaperlessBilling': paperlessbilling,
                'PaymentMethod':PaymentMethod,
                'MonthlyCharges': monthlycharges,
                'TotalCharges': totalcharges
                }
        



main()
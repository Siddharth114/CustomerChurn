import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from data_preprocessing import preprocess
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
        'Navigation',
        ("Home", 'About the model', "Contact")
    )

    if add_radio=='Home':
        st.header('Insert client information below')
        
        st.subheader("Demographic information")
        seniorcitizen = st.radio('Is the customer a Senior Citizen (>65 years of age)?', ('Yes', 'No'))
        dependents = st.radio('Does the customer have dependents who use the company\'s services?', ('Yes', 'No'))


        st.subheader("Payment information")
        tenure = st.slider('How many months has the customer stayed with the company so far?', min_value=0, max_value=72, value=0)
        contract = st.radio('What is the type of contract of the customer?', ('Month-to-month', 'One year', 'Two year'))
        paperlessbilling = st.radio('Has the customer opted in for paperless billing?', ('Yes', 'No'))
        PaymentMethod = st.radio('What is the payment method of the customer?',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
        monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
        totalcharges = st.number_input('The total amount charged to the customer',min_value=0, max_value=10000, value=0)

        st.subheader("Opted-in services")
        phoneservice = st.radio('Does the customer have phone service?', ('Yes', 'No'))
        if phoneservice=='Yes':
            mutliplelines = st.radio("Does the customer have multiple phone lines?",('Yes','No'))
        else:
            mutliplelines = 'No phone service'
        internetservice = st.radio("Does the customer have internet service?", ('DSL', 'Fiber optic', 'No'))
        if internetservice!='No':
            onlinesecurity = st.radio("Does the customer have online security?",('Yes','No'))
            onlinebackup = st.radio("Does the customer have online backup?",('Yes','No'))
            techsupport = st.radio("Does the customer have technology support?", ('Yes','No'))
            streamingtv = st.radio("Does the customer stream television?", ('Yes','No'))
            streamingmovies = st.radio("Does the customer stream movies?", ('Yes','No'))
        else:
            onlinebackup = 'No internet service'
            onlinesecurity = 'No internet service'
            techsupport = 'No internet service'
            streamingmovies = 'No internet service'
            streamingtv = 'No internet service'
        data = {
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
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.subheader('Overview of customer information is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        preprocess_df = preprocess(features_df)

        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Our model predicts that the customer is likely to leave the company.')
            else:
                st.success('Our model predicts that the customer is unlikely to leave the company.')



if __name__=='__main__':
    main()
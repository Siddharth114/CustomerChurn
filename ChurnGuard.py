import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from data_preprocessing import preprocess
import joblib
from streamlit_option_menu import option_menu

model = joblib.load(r"logistic_regression_customer_churn_classification.sav")

def main():

    nav = option_menu(None, ["Home", 'About the model', 'Contact'], 
        icons=['house', 'info', 'chat-left'], default_index=0, menu_icon='list', orientation='horizontal')


    if nav=='Home':
        st.title('ChurnGuard: Unveiling the Crystal Ball for Customer Retention')

        st.markdown('''Are you tired of losing valuable customers and revenue? Introducing our revolutionary churn prediction model powered by
        advanced machine learning algorithms. By analyzing the details of a customer, our model accurately forecasts whether a customer is likely to churn.''')
        st.markdown('''Stay one step ahead of customer attrition and take proactive measures to retain your valuable clientele.
        With our model, you can identify potential churners and implement targeted retention strategies, 
        boosting customer satisfaction and maximizing your business's bottom line. ''')
        
        st.divider()

        st.header('Insert client information below')
        
        with st.expander('Demographic information'):
            seniorcitizen = st.radio('Is the customer a Senior Citizen (>65 years of age)?', ('Yes', 'No'))
            dependents = st.radio('Does the customer have dependents who use the company\'s services?', ('Yes', 'No'))


        with st.expander("Payment information"):
            tenure = st.slider('How many months has the customer stayed with the company so far?', min_value=0, max_value=72, value=0)
            contract = st.radio('What is the type of contract of the customer?', ('Month-to-month', 'One year', 'Two year'))
            paperlessbilling = st.radio('Has the customer opted in for paperless billing?', ('Yes', 'No'))
            PaymentMethod = st.radio('What is the payment method of the customer?',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
            monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
            totalcharges = st.number_input('The total amount charged to the customer',min_value=0, max_value=10000, value=0)

        with st.expander("Opted-in services"):
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

        st.divider()

        with st.expander('Customer information summary'):
            st.dataframe(features_df)
        

        preprocess_df = preprocess(features_df)

        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Our model predicts that the customer is likely to leave the company.')
            else:
                st.success('Our model predicts that the customer is unlikely to leave the company.')

    elif nav=='About the model':
        st.title('About the model')

        st.markdown('''This potent machine learning system, created to solve the problem of customer attrition, makes use of categorical 
        client data, including demographics (age, gender, partner, dependents), financial information (monthly charges, total charges, billing method), 
        and service preferences (internet services, phone lines, TV/movie streaming). We carefully examine the distribution of 
        churn rates across various variables by thoroughly exploring the dataset.''')

        st.image(Image.open('image assets/columns.jpg'), caption='The columns of the original dataset')

        st.image(Image.open('image assets/churn distribution.jpg'), caption='Example chart of churn distribution by variable')

        st.divider()
        
        st.markdown('''We use preprocessing procedures to get the data ready for analysis, transforming appropriate columns
          into numeric types and making sure categorical feature names are clear and consistent.
          Columns such as CustomerID are not required when we apply ML algorithms, so we remove it from the dataset.
          Further, we transfer categorical features to boolean columns i.e., a column titled \'PhoneService\' which could have the values of \'Yes\' or \'No\'
          are converted into two columns titled \'PhoneService_Yes\' and \'PhoneService_No\', both of which accept boolean values.
          This makes it easier for the hyperparameter tuning of the model.''')
        
        st.image(Image.open('image assets/column correlation.jpg'), caption='Correlation heatmap of the different columns')
        
        st.divider()
        
        st.markdown(''' Next, we employ a Generalized Linear Model (GLM) to model the relationship between customer characteristics and churn probability.
        The model determines the importance of each variable, highlighting those that have a significant impact on churn rates, using their p-values.
        A high p-value denotes an unreliable (insignificant) coefficient, whereas a low p-value indicates a statistically significant coefficient.''')

        st.image(Image.open('image assets/glm results.jpg'), caption='Results of the GLM')

        st.markdown('''The question of feature importances may be answered by looking at the exponential coefficient values.
                    When one feature is modified by one unit, this number forecasts the change in value of churn.''')
        
        st.divider()

        st.markdown('''By using RFECV (Recursive Feature Elimination with Cross Validation), a powerful algorithm that finds the characteristics 
        with the most influence on the churn classification issue, we advance feature selection. This algorithm gives us the optimal number of columns 
        required for the classification problem, along with which columns in particular.''')

        st.image(Image.open('image assets/rfecv.jpg'), caption='The most important columns remaining after RFECV')

        st.divider()

        st.markdown('''We use a variety of machine learning models, such as logistic regression, decision tree classifier, Gaussian NB,
          and random forest classifier, using these ideal columns as our starting point. We conclude that logistic regression has greatest 
          precision scores, making it the best option for churn prediction after thorough analysis.''')
        
        st.image(Image.open('image assets/initial log_reg.jpg'), caption = 'Initial scores of the logistic regression model')

        st.info('''
        Accuracy score: The fraction of correctly classified samples, on a score from 0 to 1

        Precision score - The ratio of true positives to total positives, on a score from 0 to 1

        Recall score - The ratio of true positives to true positives + false negatives, on a score from 0 to 1

        f1 score - The harmonic mean of precision score and recall score, on a score from 0 to 1
        ''')

        st.divider()

        st.markdown('''We use hyperparameter optimisation to fine-tune the model's performance, maximising accuracy by optimising
          the algorithm's parameters.''')
        
        st.image(Image.open('image assets/final log_reg.jpg'), caption='The final scores of the logistic regression model')

        st.markdown('[Read More](https://github.com/Siddharth114/CustomerChurn/blob/master/main.ipynb)')

    else:
        st.title('Contact')
        st.header('Reach out to the creator')
        st.subheader('Email - 2004sid@gmail.com')
        st.subheader('[Github](https://github.com/Siddharth114)')
        st.subheader('[LinkedIn](https://www.linkedin.com/in/siddharth-m-s-566aa71b6/)')

if __name__=='__main__':
    main()
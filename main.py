import pickle
import streamlit as st
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import  streamlit_elements
from streamlit_elements import elements, mui, html




sidebar = st.sidebar

diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
heart_model= pickle.load(open('heart_model.pkl', 'rb'))



with sidebar:
    selected = option_menu(None, ["Home","Dashboard", "Diabetes Prediction", "Heart Disease Prediction"], icons=['house','bar-chart', "heart", "cast"], menu_icon="cast", default_index=0, orientation="vertical")


    # Home  Page
if selected == 'Home':
        st.subheader('Disease Prediction using ML')
        st.image('img.png', caption='Sunrise by the mountains', use_column_width=True, clamp=True, channels="RGB",
                 output_format="auto")

# Dashboard  Page
if selected == 'Dashboard':
        st.write("# DASHBOARD")
        st.write("# Diabetes model")
        st.write("This is the dataset used to train the Diabetes model")

        # loading model
        loadmodel = pickle.load(open('diabetes_model.pkl', 'rb'))
        df = pd.read_csv('diabetes.csv')
        pregn=df['Pregnancies']
        st.write(df)




        # Count of Pregnancies
        st.write("The count of Pregnancies")
        st.line_chart(pregn,use_container_width=True,height=500,width=500,x_label="Pregnancies",y_label="Count")
        avgpregnancies = df['Pregnancies'].mean()
        maxpregnancies = df['Pregnancies'].max()
        minpregnancies = df['Pregnancies'].min()
        st.write("The average number of Pregnancies is",avgpregnancies)
        st.write("The maximum number of Pregnancies is",maxpregnancies)
        st.write("The minimum number of Pregnancies is",minpregnancies)

        #glucose level count
        st.write("The count of Glucose")
        st.line_chart(df['Glucose'],use_container_width=True,height=500,width=500,x_label="Glucose",y_label="Count")
        avgglucose = df['Glucose'].mean()
        maxglucose = df['Glucose'].max()
        minglucose = df['Glucose'].min()
        st.write("The average number of Glucose is",avgglucose)
        st.write("The maximum number of Glucose is",maxglucose)
        st.write("The minimum number of Glucose is",minglucose)

        # chart for blood presure count
        st.write("The count of Blood Pressure")
        st.line_chart(df['BloodPressure'],use_container_width=True,height=500,width=500,x_label="BloodPressure",y_label="Count")
        avgbloodpressure = df['BloodPressure'].mean()
        maxbloodpressure = df['BloodPressure'].max()
        minbloodpressure = df['BloodPressure'].min()
        st.write("The average number of Blood Pressure is",avgbloodpressure)
        st.write("The maximum number of Blood Pressure is",maxbloodpressure)
        st.write("The minimum number of Blood Pressure is",minbloodpressure)

        # chart for age count
        st.write("The count of Age")
        st.line_chart(df['Age'],use_container_width=True,height=500,width=500,x_label="Age",y_label="Count")
        avgage = df['Age'].mean()
        maxage = df['Age'].max()
        minage = df['Age'].min()
        st.write("The average number of Age is",avgage)
        st.write("The maximum number of Age is",maxage)
        st.write("The minimum number of Age is",minage)

        st.write("------------------------------------------------------------------------------------------------------------------------")
        st.write("# Heart disease model")
        st.write("This is the dataset used by me to train the Heart disease model")
        # loading model
        loadmodel = pickle.load(open('heart_model.pkl', 'rb'))
        df = pd.read_csv('heart_disease_data.csv')
        st.write(df)

        st.write("Null values in dataset", df.isnull().sum())
        st.write("Information about dataset", df.info())
        st.write("Shape of dataset", df.shape)
        st.write("Statistical measures of dataset", df.describe())
        st.write("Duplicate values in dataset", df.duplicated().sum())


        # Assuming you have a DataFrame called 'df' with the data
        # Find the number of duplicates, unique rows, nulls, and other data information
        duplicates_count = df.duplicated().sum()
        unique_count = df.shape[0] - duplicates_count
        null_count = df.isnull().sum().sum()

        import pandas as pd
        import matplotlib.pyplot as plt

        # Assuming you have a DataFrame called 'df' with the data
        # Find the number of duplicates, unique rows, nulls, and other data information
        duplicates_count = df.duplicated().sum()
        unique_count = df.shape[0] - duplicates_count
        null_count = df.isnull().sum().sum()
        other_data_count = unique_count - null_count

        # Create a pie chart
        labels = ['Duplicates',  'Unique Rows']
        sizes = [duplicates_count,unique_count]
        colors = ['red', 'yellow']

        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title('Data Information')

        # Display the chart
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()




        #chart for target
        st.write("The count of Target")
        st.line_chart(df['target'],use_container_width=True,height=500,width=500,x_label="Target",y_label="Count")
        avgtarget = df['target'].mean()
        maxtarget = df['target'].max()
        mintarget = df['target'].min()
        st.write("The average number of Target is",avgtarget)
        st.write("The maximum value of Target is",maxtarget)
        st.write("The minimum value of Target is",mintarget)
        count1 = df['target'].value_counts()[1]
        st.write("Count of targets as 1 ",count1)
        count0 = df['target'].value_counts()[0]
        st.write("Count of targets as 0 ",count0)

        st.write("Chart of age ")
        st.line_chart(df['age'],use_container_width=True,height=500,width=500,x_label="Age",y_label="Count")
        avgage = df['age'].mean()
        maxage = df['age'].max()
        minage = df['age'].min()
        st.write("The average number of Age is",avgage)
        st.write("The maximum value of Age is",maxage)
        st.write("The minimum value of Age is",minage)

        st.write("Chart of sex ")
        st.line_chart(df['sex'],use_container_width=True,height=500,width=500,x_label="Sex",y_label="Count")
        avgsex = df['sex'].mean()
        maxsex = df['sex'].max()
        minsex = df['sex'].min()
        st.write("The average number of sex is",avgsex)
        st.write("The maximum value of sex is",maxsex)
        st.write("The minimum value of sex is",minsex)


        st.write("Chart of cp ")
        st.line_chart(df['cp'],use_container_width=True,height=500,width=500,x_label="Cp",y_label="Count")
        avgcp = df['cp'].mean()
        maxcp = df['cp'].max()
        mincp = df['cp'].min()
        st.write("The average number of cp is",avgcp)
        st.write("The maximum value of cp is",maxcp)
        st.write("The minimum value of cp is",mincp)



# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

        # page title
        st.title('Diabetes Prediction using ML')

        # getting the input data from the user



        Pregnancies = st.text_input('Number of Pregnancies', 0, 20, 1, 'default', 'Pregnancies')


        Glucose = st.text_input('Glucose Level')

        BloodPressure = st.text_input('Blood Pressure value')


        SkinThickness = st.text_input('Skin Thickness value')


        Insulin = st.text_input('Insulin Level')


        BMI = st.text_input('BMI value')


        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')


        Age = st.text_input('Age of the Person')

        # code for Prediction
        diab_diagnosis = ''

        # creating a button for Prediction

        if st.button('Diabetes Test Result'):

            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                          BMI, DiabetesPedigreeFunction, Age]

            user_input = [float(x) for x in user_input]

            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'

        st.success(diab_diagnosis)


                # Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

                    # page title
                    st.title('Heart Disease Prediction using ML')

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        age = st.text_input('Age')

                    with col2:
                        sex = st.text_input('Sex')

                    with col3:
                        cp = st.text_input('Chest Pain types')

                    with col1:
                        trestbps = st.text_input('Resting Blood Pressure')

                    with col2:
                        chol = st.text_input('Serum Cholestoral in mg/dl')

                    with col3:
                        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

                    with col1:
                        restecg = st.text_input('Resting Electrocardiographic results')

                    with col2:
                        thalach = st.text_input('Maximum Heart Rate achieved')

                    with col3:
                        exang = st.text_input('Exercise Induced Angina')

                    with col1:
                        oldpeak = st.text_input('ST depression induced by exercise')

                    with col2:
                        slope = st.text_input('Slope of the peak exercise ST segment')

                    with col3:
                        ca = st.text_input('Major vessels colored by flourosopy')

                    with col1:
                        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

                    # code for Prediction
                    heart_diagnosis = ''

                    # creating a button for Prediction

                    if st.button('Heart Disease Test Result'):

                        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca,
                                      thal]

                        user_input = [float(x) for x in user_input]

                        heart_prediction = heart_disease_model.predict([user_input])

                        if heart_prediction[0] == 1:
                            heart_diagnosis = 'The person is having heart disease'
                        else:
                            heart_diagnosis = 'The person does not have any heart disease'

                    st.success(heart_diagnosis)

import streamlit as st
import pandas as pd
from sympy import hyper
from classes import *
import numpy as np
import pickle

st.markdown(
'''
    # Calculate Your Risk
    Note: We will not be collecting your data so be rest assured when you key in your information. \n  
    This model might have been trained on unreliable data and the model is not perfectly accurate. \n
    This should not be taken as formal medical advice. Consult your doctor for legitimate medical advice. \n
'''
)

conversion = {
    "_gender": {
        "Male": 1.0,
        "Female": 0.0,
    },
    "_hypertension": {
        "Yes": 1.0,
        "No": 0.0,
    },
    "_heart": {
        "Yes": 1.0,
        "No": 0.0,
    },
    "_marriage": {
        "Yes": 1.0,
        "No": 0.0,
    },
    "_work": {
        "Government job": 0.0,
        "Private": 0.5,
        "Self-employed": 0.75,
        "Child": 1.0
    },
    "_residence": {
        "Rural": 0.0,
        "Urban": 1.0,
    },
    "_smoking": {
        "Formerly smoked": 0.25,
        "Never smoked": 0.5,
        "Smokes": 1.0,
    },
}

df = pd.read_csv("healthcare-dataset-stroke-data.csv", index_col=False)

def convert_numeric(val, type):
    return min(val / max(df[type]), 1.0)

def run_neural_network(inputs): 
    # Inputs are given in order
    # age, hypertension, heart, glucose, bmi, gender, marriage, work, residence, smoking
    weights1 = np.load("parameters/weights1.npy")
    biases1 = np.load("parameters/biases1.npy")
    weights2 = np.load("parameters/weights2.npy")
    biases2 = np.load("parameters/biases2.npy")
    dense1 = Layer_Dense(10, 64)
    dense2 = Layer_Dense(64, 2)
    dense1.weights = weights1
    dense1.biases = biases1
    dense2.weights = weights2
    dense2.biases = biases2
    activation = Activation_ReLU()
    softmax = Activation_Softmax()
    dense1.forward(inputs)
    activation.forward(dense1.output)
    dense2.forward(activation.output)
    softmax.forward(dense2.output)
    predictions = np.argmax(softmax.output, axis=1)

    if predictions:
        predictions = "Based on your current data, you are at risk of suffering a stroke in the near future with a " + str(round(np.max(softmax.output) * 100, 1)) + "% confidence rate."
        boolean = 1
    else:
        predictions = "Based on your current data, you are not at risk of suffering a stroke in the near future with a " + str(round(np.max(softmax.output) * 100, 1)) + "% confidence rate."
        boolean = 0

    return predictions, boolean

def run_xgb(inputs):
    xgb = pickle.load(open("parameters/XGBClassifier.sav", "rb"))
    boolean = xgb.predict(inputs)
    if boolean:
        predictions = "Based on your current data, you are at risk of suffering a stroke in the near future should your health status continue at this level."
    else:
        predictions = "Based on your current data, you are not at risk of suffering a stroke in the near future as long as your health status continues at this level."
    return predictions, boolean

def data_breakdown(gender, age, hypertension, heart, marriage, work, residence, glucose, bmi, smoking):

    if hypertension == 0.0 and heart == 0.0 and glucose < 0.5 and bmi < 0.5 and (smoking == 0.5 or smoking == 0.25):
        st.markdown(
        '''
            Your high chance of stroke is most likely caused by your age. This is an exception as a stroke is usually caused by an underlying disease or condition. In your case, all of your vitals seem to be within a healthy range. Just continue living your life like you normally would, or perhaps consider trying to live a healthier lifestyle [here](https://www.who.int/philippines/news/feature-stories/detail/20-health-tips-for-2020).
        '''
        )

    #[glucose, hypertension, heart, smoking, bmi]
    count = 0

    if count < 3 and glucose > 0.5:
        st.markdown(
        '''
            #### Your glucose level is higher than average and is a possible cause of your higher risk of stroke.
            ##### Here are some ways to lower your average glucose level:
            1. Exercise regularly
            2. Manage your carbohydrate intake
            3. Eat more fiber
            4. Drink water and stay hydrated
            5. Implement portion control
            6. Choose foods with a low glycemic index
            7. Try to manage your stress levels
            8. Monitor your blood sugar levels
            9. Get enough quality sleep
            10. Eat enough foods rich in chromium and magnesium
            11. Consider adding specific foods to your diet (Apple cider vinegar, cinnamon, berberine, fenugreek seeds)
            12. Maintain a moderate weight
            13. Eat healthy snacks more frequently
            14. Eat probiotic-rich foods
            _For more information on glucose level, visit [this website](https://www.healthline.com/nutrition/15-ways-to-lower-blood-sugar) or consult your doctor_ \n
        '''
        )
        count += 1

    if count < 3 and hypertension == 1.0:
        st.markdown(
        '''
            #### Your hypertension status could be a possible cause of your higher risk of stroke 
            ##### Here are some ways to reduce your high blood pressure without the use of medications:
            1. Lose extra weight and watch your waistline
            2. Exercise regularly
            3. Eat a healthy diet
            4. Reduce salt (sodium) in your diet
            5. Limit alcohol
            6. Quit smoking
            7. Get a good night's sleep
            8. Reduce your stress levels
            9. Monitor your blood pressure at home and get regular checkups 
            _For more information on hypertension, visit [this website](https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/in-depth/high-blood-pressure/art-20046974) or consult your doctor_ \n

        '''
        )
        count += 1

    if count < 3 and heart == 1.0:
        st.markdown(
        '''
            #### Your heart disease could be a possible cause of your higher risk of stroke
            ##### However, there is no known cure to heart disease. Fortunately, treatment can help manage the symptoms. Treatment can include:
            1. Regular exercise
            2. Quit smoking
            3. Angioplasty
            4. Surgery
            _For more information on heart disease, visit [this website](https://www.nhs.uk/conditions/coronary-heart-disease/#:~:text=Treating%20coronary%20heart%20disease%20) or consult your doctor_ \n

        '''
        )
        count += 1

    if count < 3 and smoking == 1.0:
        st.markdown(
        '''
            #### Your high risk of suffering a stoke could be due to your smoking. Do consider quitting cigarettes to lower this risk
            ##### Here are some helplines and resources to assist you in doing so:
            - [I Quit Programme](https://www.healthhub.sg/programmes/88/iquit)
            - QuitLine: 1800 438 2000
            - [Tips to quit smoking](https://www.healthhub.sg/live-healthy/598/quittips)
        '''
        )
        count += 1

    if count < 3 and bmi > 0.5:
        st.markdown(
        '''
            #### Your body mass index is higher than average and could attribute to your high risk of suffering a stroke
            ##### Here are some ways to reduce your BMI:
            1. Balance your food choices
            2. Watch what you eat
            3. Get at least 150 minutes of physical activity weekly
            4. Build up your strength
            5. Have regular meals
            _For more information on heart disease, visit [this website](https://www.healthhub.sg/live-healthy/408/Healthy%20Weight%20Loss) or consult your doctor_ \n 
        '''
        )
        count += 1
            
        

with st.form("my_form"):
    gender = conversion["_gender"][st.radio("Select your gender", ["Male", "Female"])]
    age = convert_numeric(st.number_input("Enter your age", min_value=0, step=1), "age")
    hypertension = conversion["_hypertension"][st.radio("Do you have hypertension?", ["Yes", "No"])]
    heart = conversion["_heart"][st.radio("Do you have heart disease?", ["Yes", "No"])]
    marriage = conversion["_marriage"][st.radio("Have you been married?", ["Yes", "No"])]
    work = conversion["_work"][st.radio("What kind of job do you have?", ["Government job", "Private", "Self-employed", "Child"])]
    residence = conversion["_residence"][st.radio("What is your residence type?", ["Rural", "Urban"])]
    glucose = convert_numeric(st.number_input("What is your average glucose level? (in mg / dL)", min_value=0), "avg_glucose_level")
    bmi = convert_numeric(st.number_input("What is your BMI (Body Mass Index)?", min_value=0.0, step=0.1), "bmi")
    smoking = conversion["_smoking"][st.radio("What is your smoking status?", ["Formerly smoked", "Never smoked", "Smokes"])]
    
    submitted = st.form_submit_button("Submit My Data")

    inputs = gender, age, hypertension, heart, marriage, work, residence, glucose, bmi, smoking
    
if submitted:
    #stroke, boolean = run_neural_network(np.array([inputs]))
    stroke, boolean = run_xgb(np.array([inputs]))
    st.title(stroke)

    if boolean:
        data_breakdown(gender, age, hypertension, heart, marriage, work, residence, glucose, bmi, smoking)
    else:
        st.write("Do continue keeping up with your healthy lifestyle to enjoy a stroke-free life!")


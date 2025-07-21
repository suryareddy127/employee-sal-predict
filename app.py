# app/app.py

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="💵Salary Class Predictor", layout="centered")

# 🎯 Load model
model_path = os.path.join("models", "model.pkl")
if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please train and save pipeline first.")
    st.stop()

pipeline = joblib.load(model_path)

# 🚀 introduction
st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #f8fafc 0%, #e0e7ff 100%);
        }
        .title {
            text-align: center;
            font-size: 44px;
            font-weight: 900;
            color: #8B0000;
            margin-top: 17px;
            letter-spacing: 1.7px;
        }
        .subtitle {
            text-align: center;
            font-size: 17px;
            color: #4B0082;
            margin-bottom: 30px;
        }
    </style>
    <div class="title"><b>👾Employee Salary Predictor</b></div>
    <div class="subtitle">Upload CSV or use form below to get predictions!</div>
""", unsafe_allow_html=True)

# 📂 CSV Upload
st.header("🗂️ Upload Dataset for Prediction")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.dataframe(input_df.head())

        if 'capital_diff' not in input_df.columns:
            if 'capital-gain' in input_df.columns and 'capital-loss' in input_df.columns:
                input_df['capital_diff'] = input_df['capital-gain'] - input_df['capital-loss']
        if 'experience_level' not in input_df.columns:
            if 'age' in input_df.columns:
                input_df['experience_level'] = pd.cut(
                    input_df['age'],
                    bins=[15, 25, 35, 50, 65, 100],
                    labels=['Entry', 'Junior', 'Mid', 'Senior', 'Executive']
                )

        predictions = pipeline.predict(input_df)
        input_df["Salary Class"] = ["<=50K" if pred == 0 else ">50K" for pred in predictions]

        st.success("✅ Predictions Done!")
        st.dataframe(input_df)
    except Exception as e:
        st.error(f"🚫 Could not process file: {e}")

st.markdown("---")

st.header("🔎Predict Salary Class from User Input")

# Sidebar for user input

if uploaded_file:
    st.sidebar.header("Enter User Data for Prediction")
    with st.sidebar.form(key="manual_form"):
        age = st.slider("Age", 18, 70, value=30)
        workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"])
        education_num = st.slider("Education Level (Num)", 1, 16, value=9)
        occupation = st.selectbox("Occupation", ["Exec-managerial", "Craft-repair", "Sales", "Tech-support", "Others"])
        relationship = st.selectbox("Relationship", ["Husband", "Wife", "Own-child", "Not-in-family", "Unmarried"])
        race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        capital_gain = st.number_input("Capital Gain", 0, 99999, value=0)
        capital_loss = st.number_input("Capital Loss", 0, 9999, value=0)
        hours_per_week = st.slider("Hours per Week", 1, 100, value=40)
        native_country = st.selectbox("Native Country", ["United-States", "India", "Germany", "Mexico", "Philippines"])
        submit_btn = st.form_submit_button("Predict")

    # Main page: show prediction result after submit
    if submit_btn:
        input_dict = {
            "age": age,
            "workclass": workclass,
            "education-num": education_num,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "gender": gender,
            "capital-gain": capital_gain,
            "capital-loss": capital_loss,
            "hours-per-week": hours_per_week,
            "native-country": native_country
        }

        # Add missing columns with default values
        input_dict["education"] = "Bachelors"
        input_dict["marital-status"] = "Never-married"
        input_dict["capital_diff"] = capital_gain - capital_loss
        input_dict["educational-num"] = education_num
        input_dict["experience_level"] = pd.cut(
            [age],
            bins=[15, 25, 35, 50, 65, 100],
            labels=["Entry", "Junior", "Mid", "Senior", "Executive"]
        )[0]

        input_df = pd.DataFrame([input_dict])
        try:
            prediction = pipeline.predict(input_df)[0]
            result = ">50K" if prediction == 1 else "<=50K"
            st.sidebar.markdown("---")
            st.sidebar.header("🎯 Predicted Salary Class for Input:")
            st.sidebar.subheader(result)

            # Show result in center of main page
            st.markdown("""
                <div style='display: flex; justify-content: center; align-items: center; height: 100px;'>
                    <span style='font-size: 32px; font-weight: bold; color: orange;'>🎯 Predicted Salary Class: {}</span>
                </div>
            """.format(result), unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error(f"🚫 Prediction Failed: {e}")

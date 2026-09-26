import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    with open("pipe.pkl", "rb") as f:
        model = pickle.load(f)
    return model

pipe = load_model()

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details below to predict survival.")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=30.0, step=1.0)

with col2:
    sibsp = st.number_input("Siblings/Spouses aboard (SibSp)", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parents/Children aboard (Parch)", min_value=0, max_value=10, value=0, step=1)
    embarked = st.selectbox("Port of Embarkation (Embarked)", ["S", "C", "Q"])


input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked]
})

st.subheader("Input Summary")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Survival"):
    try:
        prediction = pipe.predict(input_df)[0]

        proba = None
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(input_df)[0]

        if prediction == 1:
            st.success("✅ This passenger would likely SURVIVE.")
        else:
            st.error("❌ This passenger would likely NOT SURVIVE.")

        if proba is not None:
            st.write(f"**Survival probability:** {proba[1]*100:.2f}%")
            st.write(f"**Death probability:** {proba[0]*100:.2f}%")
            st.progress(float(proba[1]))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "This usually means the input column names/order or data types "
            "don't match what the pipeline was trained on. Double-check "
            "pipe.pkl's expected feature format."
        )
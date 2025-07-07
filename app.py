# Save this to a file programmatically
import streamlit as st
import pandas as pd
import joblib

# load model
model = joblib.load("logistic_model.pkl")

st.title("🚢 Titanic Survival Prediction")

st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3], index=2)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 30)
fare = st.sidebar.slider("Fare", 0, 600, 50)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])
sibsp = st.sidebar.slider("Number of Siblings/Spouses aboard (SibSp)", 0, 8, 0)
parch = st.sidebar.slider("Number of Parents/Children aboard (Parch)", 0, 6, 0)

# encode
sex_encoded = 0 if sex == "male" else 1
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_map[embarked]

# match the same 7-feature input used during training
input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_encoded],
    "Age": [age],
    "Fare": [fare],
    "Embarked": [embarked_encoded],
    "SibSp": [sibsp],
    "Parch": [parch]
})

st.subheader("Passenger Features")
st.write(input_df)

if st.button("Predict Survival"):
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]
    result = "Survived ✅" if pred == 1 else "Did not Survive ❌"

    st.subheader("Prediction Result")
    st.write(f"**{result}** (probability of survival: {pred_proba:.2f})")

    if pred == 1:
        st.balloons()

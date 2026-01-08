import os
import streamlit as st
import requests
from dotenv import load_dotenv


# Environment configuration
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")

# Streamlit
st.set_page_config(page_title="TITANIC", layout="centered")
st.title("TITANIC")

# Load Model
# @st.cache_resource
# def load_model():
#     return joblib.load("../data/titanic_model.pkl")
# model = load_model()

st.subheader("Passenger Features")
c1, c2, c3 = st.columns(3)

with c1:
    PassengerName = st.text_input("Passenger Name", value="Kundjanasith Thonglek")
    Pclass = st.selectbox("Pclass", [1, 2, 3], index=1)
    Sex = st.selectbox("Sex", ["male", "female"], index=1)

with c2:
    Age = st.number_input("Age", min_value=0, max_value=100, value=28, step=1)
    SibSp = st.number_input("SibSp", min_value=0, max_value=10, value=0, step=1)
    Parch = st.number_input("Parch", min_value=0, max_value=10, value=0, step=1)

with c3:
    Ticket = st.text_input("Ticket", value="AIENABLE 888")
    Fare = st.number_input("Fare", min_value=0.0, value=26.0, step=0.1)
    Embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)

if st.button("Predict"):
    try:
        # Ticket feature preprocessing
        split_ticket_val = Ticket.split(" ")
        TicketNumber = split_ticket_val.str[-1]
        if len(split_ticket_val) > 1:
            TicketPrefix = "".join(split_ticket_val.str[:-1])
        else:
            TicketPrefix = ""
        HasPrefix = 1 if TicketPrefix else 0
        TicketLength = len(str(TicketNumber))
        TicketIsLine = 1 if str(TicketNumber).upper() == "LINE" else 0
        TicketNumber = int(TicketNumber) if TicketNumber.isdigit() else 0

        # payload
        payload = {
            "Pclass": Pclass,
            "Sex": Sex,
            "Age": Age,
            "SibSp": SibSp,
            "Parch": Parch,
            "Fare": Fare,
            "Embarked": Embarked,
            "HasPrefix": HasPrefix,
            "TicketNumber": TicketNumber,
            "TicketIsLine": TicketIsLine,
            "TicketLength": TicketLength
        }

        endpoint = f"{BACKEND_URL}/predict"
        resp = requests.post(endpoint, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        pred = data["prediction"]  # "Survived" or "Not Survived"
        prob = data["prob_survive"]

        emoji = "🛟" if pred == "Survived" else "💀"*3
        st.subheader(emoji)
        st.write(f"Prediction: **{pred}**")
        st.write(f"Probability of survival: **{prob}**")
    except Exception as e:
        st.error(f"Error calling Backend API: {e}")

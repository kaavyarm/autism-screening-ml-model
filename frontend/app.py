import streamlit as st
import requests

st.set_page_config(page_title="Autism Screening", layout="centered")

st.title("Autism Screening Tool")
st.write(
    "Answer the following questions. This tool uses a machine learning model "
    "to estimate likelihood of autism traits. It is **not a diagnosis**."
)

st.subheader("Behavioral Screening (AQ-10 Inspired)")

questions = [
    "I often notice small sounds when others do not",
    "I usually concentrate more on the whole picture, rather than small details",
    "I find it easy to do more than one thing at once",
    "If there is an interruption, I can switch back very quickly",
    "I find it easy to read between the lines when someone is talking",
    "I know how to tell if someone listening to me is getting bored",
    "I find it easy to work out what someone is thinking or feeling",
    "I like to collect information about categories of things",
    "I find it easy to work out people’s intentions",
    "I find it easy to imagine what someone else might be thinking"
]

# AQ-10 scoring directions
# True = reverse scored
reverse_questions = {
    1: False,
    2: True,
    3: True,
    4: True,
    5: True,
    6: True,
    7: True,
    8: False,
    9: True,
    10: True
}

aq_scores = []

for i, q in enumerate(questions, start=1):
    response = st.radio(
        f"{i}. {q}",
        ["Yes", "No"],
        key=f"q{i}"
    )

    if reverse_questions[i]:
        score = 0 if response == "Yes" else 1
    else:
        score = 1 if response == "Yes" else 0

    aq_scores.append(score)

st.subheader("Basic Information")

age = st.number_input("Age", min_value=1, max_value=100, value=18)

gender = st.selectbox("Gender", ["Male", "Female"])
gender_val = 1 if gender == "Male" else 0

jundice = st.selectbox("History of Jaundice at Birth?", ["Yes", "No"])
jundice_val = 1 if jundice == "Yes" else 0

austim = st.selectbox("Family History of Autism?", ["Yes", "No"])
austim_val = 1 if austim == "Yes" else 0

if st.button("Run Screening"):

    aq_total = sum(aq_scores)

    payload = {
        "A1_Score": aq_scores[0],
        "A2_Score": aq_scores[1],
        "A3_Score": aq_scores[2],
        "A4_Score": aq_scores[3],
        "A5_Score": aq_scores[4],
        "A6_Score": aq_scores[5],
        "A7_Score": aq_scores[6],
        "A8_Score": aq_scores[7],
        "A9_Score": aq_scores[8],
        "A10_Score": aq_scores[9],
        "age": age,
        "gender": gender_val,
        "jundice": jundice_val,
        "austim": austim_val
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    result = response.json()

    prediction = result["prediction"]
    probability = result["probability"]

    st.subheader("Screening Result")

    st.write(f"AQ Score: **{aq_total}/10**")
    st.write(f"Model Risk Score: **{probability:.2f}**")

    if probability < 0.3:
        st.success("Low likelihood of autism traits")
    elif probability < 0.7:
        st.warning("Moderate likelihood — consider further screening")
    else:
        st.error("High likelihood — consider professional evaluation")

    st.caption("This is a screening tool, not a medical diagnosis.")
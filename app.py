import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("data/titanic.csv")

# Drop Unnecessary Columns
drop_columns = ["PassengerId", "Name", "Ticket", "Cabin"]

for col in drop_columns:
    if col in data.columns:
        data = data.drop(col, axis=1)

# Handle Missing Values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Encode Categorical Columns
label_encoder = LabelEncoder()

for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])

# Features and Target
target = "Survived"

X = data.drop(target, axis=1)
y = data[target]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load Model
model = joblib.load("models/adaboost_classifier.pkl")

# Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("🚢 Titanic Survival Prediction - AdaBoost Classifier")

# Display Accuracy
st.subheader("Model Accuracy")

st.success(f"Accuracy : {accuracy:.2f}")

# User Inputs
st.subheader("Enter Passenger Details")

Pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

Sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

Age = st.slider(
    "Age",
    1,
    80,
    25
)

SibSp = st.number_input(
    "Siblings / Spouses Aboard",
    0,
    10,
    0
)

Parch = st.number_input(
    "Parents / Children Aboard",
    0,
    10,
    0
)

Fare = st.number_input(
    "Fare",
    0.0,
    600.0,
    50.0
)

Embarked = st.selectbox(
    "Embarked",
    ["C", "Q", "S"]
)

# Encode Inputs
Sex = 1 if Sex == "male" else 0

embarked_map = {
    "C": 0,
    "Q": 1,
    "S": 2
}

Embarked = embarked_map[Embarked]

# Input Data
input_data = pd.DataFrame({
    "Pclass": [Pclass],
    "Sex": [Sex],
    "Age": [Age],
    "SibSp": [SibSp],
    "Parch": [Parch],
    "Fare": [Fare],
    "Embarked": [Embarked]
})

# Prediction
if st.button("Predict Survival"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
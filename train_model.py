import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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

# Base Model
base_model = DecisionTreeClassifier(max_depth=1)

# AdaBoost Classifier
model = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "models/adaboost_classifier.pkl")

print("Model trained and saved successfully!")
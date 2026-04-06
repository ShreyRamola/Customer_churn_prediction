import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Step 1: Create Dataset
# -----------------------------
data = pd.DataFrame({
    'Age': [22, 25, 47, 52, 46, 56, 23, 34, 42, 50],
    'Monthly_Charges': [200, 150, 300, 350, 280, 400, 120, 220, 270, 320],
    'Contract_Type': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year',
                      'Two year', 'Month-to-month', 'One year', 'Two year', 'Month-to-month'],
    'Tenure': [1, 12, 24, 3, 15, 30, 2, 10, 20, 5],
    'Churn': ['Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes']
})

print("Dataset:\n", data)

# -----------------------------
# Step 2: Encode Data
# -----------------------------
le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# -----------------------------
# Step 3: Features & Target
# -----------------------------
X = data.drop('Churn', axis=1)
y = data['Churn']

# -----------------------------
# Step 4: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 5: Train Naive Bayes Model
# -----------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -----------------------------
# Step 6: Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 7: Evaluation
# -----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
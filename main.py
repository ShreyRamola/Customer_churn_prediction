import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "Age": np.random.randint(18, 70, n),
    "Tenure": np.random.randint(1, 72, n),
    "Monthly": np.random.uniform(20, 120, n),
    "Total": np.random.uniform(100, 8000, n),
    "Contract": np.random.choice([0, 1, 2], n),
    "Internet": np.random.choice([0, 1, 2], n),
    "Support": np.random.choice([0, 1], n),
    "Senior": np.random.choice([0, 1], n),
})

df["Churn"] = (np.random.rand(n) < (0.05 + 0.30 * (df["Contract"] == 0) + 
    0.10 * (df["Tenure"] < 12) + 0.10 * (df["Monthly"] > 80) + 
    0.05 * (df["Support"] == 0))).astype(int)

X, y = df.drop("Churn", axis=1), df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(100, random_state=42).fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Data: {n} samples | Churn: {y.mean()*100:.1f}%")
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}\n")

print("Top 3 Features:")
for i in np.argsort(model.feature_importances_)[-3:]:
    print(f"  {X.columns[i]}: {model.feature_importances_[i]:.4f}")

new = scaler.transform(pd.DataFrame([[35, 5, 95, 475, 0, 1, 0, 0]], columns=X.columns))
pred, prob = model.predict(new)[0], model.predict_proba(new)[0][1]
print(f"\nCustomer: {'CHURN' if pred else 'NO CHURN'} ({prob*100:.1f}%)")
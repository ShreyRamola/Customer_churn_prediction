📊 Customer Churn Prediction using Naive Bayes

🔹 Overview

This project demonstrates a simple Machine Learning model to predict customer churn using the Naive Bayes algorithm.

Customer churn refers to whether a customer leaves a service or continues using it. This is a binary classification problem.

---

🔹 Algorithm Used

- Naive Bayes (Gaussian Naive Bayes)

Why Naive Bayes?

- Based on probability and Bayes Theorem
- Fast and efficient
- Works well with small datasets
- Easy to implement and explain

---

🔹 Dataset

- The dataset is manually created inside the code
- Contains features like:
  - Age
  - Monthly Charges
  - Contract Type
  - Tenure
- Target variable:
  - Churn (Yes/No)

---

🔹 Project Workflow

1. Create dataset using Pandas
2. Encode categorical data using Label Encoding
3. Split dataset into training and testing sets
4. Train model using Gaussian Naive Bayes
5. Predict churn values
6. Evaluate model using:
   - Accuracy
   - Confusion Matrix
   - Classification Report

---

🔹 Requirements

Install the following libraries before running:

pip install pandas numpy scikit-learn

---

🔹 How to Run

1. Clone the repository

git clone https://github.com/your-username/customer-churn-naive-bayes.git

2. Navigate to the project folder

cd customer-churn-naive-bayes

3. Run the Python file

python churn_model.py

---

🔹 Output

The model will display:

- Accuracy score
- Confusion matrix
- Precision, recall, and F1-score

---

🔹 Key Concepts

- Classification
- Naive Bayes Algorithm
- Data Preprocessing
- Model Evaluation

---

🔹 Future Improvements

- Use a real-world dataset
- Add GUI (Tkinter / Streamlit)
- Compare with other algorithms like:
  - Logistic Regression
  - Decision Tree
  - SVM

---

🔹 Conclusion

This project shows how Naive Bayes can be used for predicting customer churn in a simple and efficient way. It is suitable for beginners and academic purposes.

---

👨‍💻 Author

Shrey Ramola
B.Tech CSE Student
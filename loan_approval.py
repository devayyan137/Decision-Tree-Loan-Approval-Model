import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = {
    "Age": [22, 25, 47, 52, 46, 56, 24, 33, 38, 44, 29, 41, 50, 37, 28],
    "Income": [25000, 30000, 80000, 90000, 75000, 100000, 28000, 45000, 60000, 70000, 40000, 65000, 85000, 55000, 35000],
    "LoanAmount": [5000, 7000, 20000, 25000, 18000, 30000, 6000, 12000, 15000, 17000, 10000, 16000, 22000, 14000, 8000],
    "CreditScore": [600, 650, 750, 720, 700, 780, 620, 680, 710, 730, 660, 705, 760, 690, 640],
    "LoanApproved": [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]}
df = pd.DataFrame(data)
X = df.drop("LoanApproved", axis=1)
y = df["LoanApproved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
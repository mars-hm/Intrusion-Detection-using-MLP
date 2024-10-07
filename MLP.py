import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

df_train = pd.read_csv(r"D:\M.Sc CS\AI\Intrusion Detection using Multi Layer Perceptron\NSL-KDD\train.csv", header=None)  # Example filename for training
df_train.head()

missing_values = df_train.isnull().sum()
print(df_train.shape)
df_train = df_train.dropna()  # Or fill missing values: df.fillna(0)
print(df_train.shape)

df_train.nunique()

from sklearn.preprocessing import LabelEncoder
X = df_train.iloc[:, :-1].copy()  
for col in X.columns:
    if X[col].dtype == 'object' or X[col].apply(lambda x: isinstance(x, (str, int))).any():
        X.loc[:, col] = X[col].astype(str)
        X.loc[:, col] = LabelEncoder().fit_transform(X[col])

X = X.astype('float')
print("Encoded Features\n", X.head())

y1 = df_train.iloc[:, -1].values 
y = LabelEncoder().fit_transform(y1)
print("\nLabels\n",y)

chi2_selector = SelectKBest(chi2, k=20)  # Change k based on your needs
X_kbest = chi2_selector.fit_transform(X, y)
X_kbest.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # Define hidden layers
    activation='relu',  # Activation function
    solver='adam',  # Optimizer
    max_iter=20,  # Number of iterations
    random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

mname='mlp_ids.pkl'
joblib.dump(model, mname)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt


np.random.seed(42)


data = pd.read_csv('/Users/ipekucar/Desktop/diabetes.csv')


print("Step 1: Analyze the Original Dataset")
print("Original Dataset (First 5 Rows):")
print(data.head())
print()


X = data.iloc[:, :-1]
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Class Distribution in Test Set (y_test):")
print(y_test.value_counts())
print()


svm_model_original = SVC()
svm_model_original.fit(X_train, y_train)
y_pred_svm_original = svm_model_original.predict(X_test)
cm_svm_original = confusion_matrix(y_test, y_pred_svm_original)
accuracy_svm_original = accuracy_score(y_test, y_pred_svm_original)

print("Confusion Matrix for SVM (Original Dataset):")
print(pd.DataFrame(cm_svm_original, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
print("Accuracy for SVM (Original Dataset):", accuracy_svm_original)
print()


knn_model_original = KNeighborsClassifier()
knn_model_original.fit(X_train, y_train)
y_pred_knn_original = knn_model_original.predict(X_test)
cm_knn_original = confusion_matrix(y_test, y_pred_knn_original)
accuracy_knn_original = accuracy_score(y_test, y_pred_knn_original)

print("Confusion Matrix for KNN (Original Dataset):")
print(pd.DataFrame(cm_knn_original, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
print("Accuracy for KNN (Original Dataset):", accuracy_knn_original)
print()

print("Step 2: Replace Some Values with NaN")
columns_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[columns_with_zeros] = data[columns_with_zeros].replace(0, np.nan)

print("Number of NaN values in each column before filling:")
print(data[columns_with_zeros].isna().sum())
print()

print("Step 3: Replace NaN Values with Column Means")
data_filled = data.copy()
data_filled[columns_with_zeros] = data_filled[columns_with_zeros].fillna(data_filled[columns_with_zeros].mean())


print("Number of NaN values in each column after filling:")
print(data_filled[columns_with_zeros].isna().sum())
print()

print("Dataset After Replacing NaN with Column Means (First 5 Rows):")
print(data_filled.head())
print()


print("Step 4: Classification After Replacing NaN with Column Means")
X_filled = data_filled.iloc[:, :-1]
y_filled = data_filled.iloc[:, -1]


smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


svm_model_balanced = SVC()
svm_model_balanced.fit(X_train_balanced, y_train_balanced)
y_pred_svm_balanced = svm_model_balanced.predict(X_test)
cm_svm_balanced = confusion_matrix(y_test, y_pred_svm_balanced)
accuracy_svm_balanced = accuracy_score(y_test, y_pred_svm_balanced)

print("Confusion Matrix for SVM (After Balancing with SMOTE):")
print(pd.DataFrame(cm_svm_balanced, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
print("Accuracy for SVM (After Balancing with SMOTE):", accuracy_svm_balanced)
print()


knn_model_balanced = KNeighborsClassifier()
knn_model_balanced.fit(X_train_balanced, y_train_balanced)
y_pred_knn_balanced = knn_model_balanced.predict(X_test)
cm_knn_balanced = confusion_matrix(y_test, y_pred_knn_balanced)
accuracy_knn_balanced = accuracy_score(y_test, y_pred_knn_balanced)

print("Confusion Matrix for KNN (After Balancing with SMOTE):")
print(pd.DataFrame(cm_knn_balanced, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
print("Accuracy for KNN (After Balancing with SMOTE):", accuracy_knn_balanced)
print()


print("Additional Metrics for SVM (After Balancing with SMOTE):")
print("Precision (SVM):", precision_score(y_test, y_pred_svm_balanced))
print("Recall (SVM):", recall_score(y_test, y_pred_svm_balanced))
print("F1-Score (SVM):", f1_score(y_test, y_pred_svm_balanced))
print()


print("Additional Metrics for KNN (After Balancing with SMOTE):")
print("Precision (KNN):", precision_score(y_test, y_pred_knn_balanced))
print("Recall (KNN):", recall_score(y_test, y_pred_knn_balanced))
print("F1-Score (KNN):", f1_score(y_test, y_pred_knn_balanced))
print()


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

plot_confusion_matrix(cm_svm_balanced, "Confusion Matrix for SVM (After Balancing with SMOTE)")


plot_confusion_matrix(cm_knn_balanced, "Confusion Matrix for KNN (After Balancing with SMOTE)")

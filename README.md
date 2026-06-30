# Diabetes Prediction (Data Mining Project)

A binary classification project on the **Pima Indians Diabetes** dataset (`diabetes.csv`, Kaggle), built in **three iterative versions** that progressively strengthen the data-mining workflow, going from a naive baseline to proper handling of missing values, class imbalance, and evaluation.

## Dataset
- `data/diabetes.csv`: 768 patient records, 8 clinical features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) and a binary `Outcome` (diabetes / no diabetes).

## Versions
Each stage lives in its own folder with its own README and saved output:

| Version | What it adds |
|---|---|
| [`initial-version/`](initial-version) | Baseline **SVM** and **KNN** classifiers. Preprocessing: biologically impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI replaced with `NaN` and filled by column mean. 80/20 train-test split, confusion matrix and accuracy. |
| [`optimized-version/`](optimized-version) | Adds **precision, recall and F1-score**, confusion-matrix **heatmaps**, and cleaner preprocessing. |
| [`final-version/`](final-version) | Adds **SMOTE** to handle class imbalance; evaluates SVM and KNN across the original, preprocessed and balanced datasets with the full metric set. |

## Techniques
- **Models:** Support Vector Machine (SVM), K-Nearest Neighbors (KNN)
- **Preprocessing:** missing-value detection (invalid zeros set to `NaN`), mean imputation, train-test split
- **Class imbalance:** SMOTE oversampling
- **Evaluation:** accuracy, precision, recall, F1-score, confusion matrices (heatmaps)

## Results (final version, balanced SMOTE dataset)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM | 0.69 | 0.56 | 0.67 | 0.61 |
| KNN | 0.64 | 0.49 | 0.73 | 0.59 |

Full per-stage outputs are saved in each version's `*-output.txt`.

## Run
```bash
pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
python final-version/DataMining_Final.py
```

## Tech stack
Python, scikit-learn, imbalanced-learn (SMOTE), pandas, NumPy, matplotlib / seaborn

## Notes and next steps
Hyperparameter tuning is not yet implemented, so these are **baseline** results. Natural next steps: cross-validation, tuned SVM/KNN, and trying tree-based or ensemble models (Random Forest, gradient boosting).

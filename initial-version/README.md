# Data Mining Project - Initial Version

This project analyzes a diabetes dataset using machine learning classification models: Support Vector Machine (SVM) and K-Nearest Neighbors (KNN). Missing values are handled by replacing zeros with `NaN` and filling them with column means.

---
## Dataset

The dataset used in this project is publicly available and can be downloaded from [Kaggle](https://www.kaggle.com/datasets/saurabh00007/diabetescsv).

### Steps to Use the Dataset
1. Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/saurabh00007/diabetescsv).
2. Download the file `diabetes.csv`.
3. Place the file in the `data` folder within the project directory: /data/diabetes.csv
4. Run the code as described in the "How to Run" section.

## Steps in the Code

1. **Original Dataset Analysis:**
   - Load the dataset (`diabetes.csv`) and display the first 5 rows.
   - Split the data into training and testing sets (80% training, 20% testing).
   - Train and evaluate SVM and KNN models on the original dataset.
   - Display confusion matrices and accuracy scores.

2. **Handle Missing Values:**
   - Replace zeros in specific columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with `NaN`.
   - Fill `NaN` values with the mean of each column.

3. **Cleaned Dataset Analysis:**
   - Train and evaluate SVM and KNN models on the cleaned dataset.
   - Display updated confusion matrices and accuracy scores.

---

## Dataset Details

- **Features:**
  - `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
- **Target Variable:**
  - `Outcome`: Binary (1 = Diabetes, 0 = No Diabetes).

---

### How to Run

1. Download the dataset as explained in the "Dataset" section.
2. Place the dataset in the `/data` folder.
3. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn

4. Run the script:
   python DataMining_Initial.py

## Outputs

- All outputs for this version are saved in the initial-version/inital-version-output.txt  file. Confusion matrices and accuracy scores are provided for:

   1. Original Dataset Analysis (Before Cleaning)
   2. Cleaned Dataset Analysis (After Replacing NaN)
   -  Please refer to the output file for detailed results.

## Limitations of This Version

   1. Class imbalance is not addressed in this version.
   2. No additional metrics like precision, recall, or F1-score are calculated.
   3. No hyperparameter tuning for SVM or KNN is implemented.

## Next Steps

- In the optimized version, we will: 

  1. Calculate additional performance metrics such as precision, recall, and F1-score.
  2. Visualize confusion matrices using heatmaps for better interpretability.
  3. Improve code structure and clarity by separating preprocessing and evaluation steps.





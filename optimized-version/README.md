# Data Mining Project - Optimized Version

This project builds upon the initial version by implementing additional metrics and visualizations to enhance the analysis of a diabetes dataset. The primary improvements include calculating precision, recall, and F1-score, as well as generating heatmaps for confusion matrices.

---

## Dataset

The dataset used in this project is publicly available and can be downloaded from [Kaggle](https://www.kaggle.com/datasets/saurabh00007/diabetescsv).

### Steps to Use the Dataset
1. Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/saurabh00007/diabetescsv).
2. Download the file `diabetes.csv`.
3. Place the file in the `data` folder within the project directory: /data/diabetes.csv
4. Run the code as described in the "How to Run" section.


## Improvements Over Initial Version

1. **Additional Metrics:**
   - Precision, Recall, and F1-Score are calculated for both SVM and KNN classifiers.

2. **Visualization:**
   - Heatmaps of confusion matrices are generated for better interpretability.

3. **Code Organization:**
   - The dataset is preprocessed more clearly, and evaluation steps are systematically separated.

---

## Steps in the Code

1. **Original Dataset Analysis:**
   - Load the dataset (`diabetes.csv`) and display the first 5 rows.
   - Split the data into training and testing sets (80% training, 20% testing).
   - Train and evaluate SVM and KNN models on the original dataset.
   - Display confusion matrices, accuracy scores, and class distributions.

2. **Handle Missing Values:**
   - Replace zeros in specific columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with `NaN`.
   - Fill `NaN` values with the mean of each column.

3. **Cleaned Dataset Analysis:**
   - Retrain and evaluate SVM and KNN models on the cleaned dataset.
   - Calculate additional metrics: Precision, Recall, and F1-Score.
   - Generate heatmaps of confusion matrices.

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
   python DataMining_Optimized.py


## Outputs
- All outputs for this version are saved in the outputs/optimized-version-output.txt file. Metrics include:

  1. **Original Dataset Analysis**
  Confusion matrices and accuracy scores for SVM and KNN.
  2. **Cleaned Dataset Analysis**
  Confusion matrices, accuracy scores, precision, recall, and F1-scores for SVM and KNN.
  3. **Heatmaps**
  Heatmaps of confusion matrices are displayed during execution.


## Limitations of This Version
  1. Class imbalance is not addressed.
  2. No hyperparameter tuning for SVM or KNN is implemented.
  3. Model interpretability and feature importance are not explored.


## Next Steps
- In the final version, we will:

  1. Address class imbalance using SMOTE or similar techniques.
  2. Perform hyperparameter tuning for SVM and KNN models.
  3. Explore advanced visualizations and feature importance analysis.   

   


# Data Mining Project - Final Version

This is the final version of the Data Mining Project. It builds upon the previous versions by addressing class imbalance using SMOTE and improving model evaluation with additional metrics and hyperparameter tuning.

---
## Dataset

The dataset used in this project is publicly available and can be downloaded from [Kaggle](https://www.kaggle.com/datasets/saurabh00007/diabetescsv).

### Steps to Use the Dataset
1. Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/saurabh00007/diabetescsv).
2. Download the file `diabetes.csv`.
3. Place the file in the `data` folder within the project directory: /data/diabetes.csv
4. Run the code as described in the "How to Run" section.

## Key Features

1. **SMOTE Implementation:**
   - Class imbalance in the dataset is addressed using SMOTE (Synthetic Minority Oversampling Technique).

2. **Additional Metrics:**
   - Metrics such as Precision, Recall, and F1-Score are calculated for both SVM and KNN classifiers.

3. **Visualizations:**
   - Confusion matrices are visualized as heatmaps for better interpretability.

4. **Code Refinement:**
   - Improved structure and comments for clarity and better maintainability.

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
   - Generate confusion matrices and accuracy scores.

4. **SMOTE Implementation:**
   - Apply SMOTE to balance the training data for fair evaluation.
   - Train and evaluate SVM and KNN models on the balanced dataset.
   - Calculate additional metrics: Precision, Recall, and F1-Score.

5. **Heatmap Visualizations:**
   - Generate heatmaps of confusion matrices for both SVM and KNN classifiers after balancing with SMOTE.

---

### How to Run

1. Download the dataset as explained in the "Dataset" section.
2. Place the dataset in the `/data` folder.
3. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib


4. Run the script : 
   python DataMining_Final.py
   
## Outputs
- All outputs for this version are saved in the outputs/final-version-output.txt file. Metrics include:

1. **Original Dataset Analysis**
   - Confusion matrices and accuracy scores for SVM and KNN.
2. **Cleaned Dataset Analysis**
   - Confusion matrices and accuracy scores for SVM and KNN after filling missing values.
3. **Balanced Dataset Analysis**
   - Confusion matrices, accuracy scores, precision, recall, and F1-scores for SVM and KNN after applying SMOTE.
4. **Heatmaps**
   - Heatmaps of confusion matrices are displayed during execution.

## Example Metrics (Balanced Dataset)

   **SVM Metrics:**
   - Accuracy: 0.6948
   - Precision: 0.5606
   - Recall: 0.6727
   - F1-Score: 0.6116
   **KNN Metrics:**
   - Accuracy: 0.6364
   - Precision: 0.4938
   - Recall: 0.7273
   - F1-Score: 0.5882

## Limitations of This Version
   1. Hyperparameter tuning is not fully implemented for SVM and KNN models.
   2. Feature importance analysis is not explored.
   3. Advanced visualizations like ROC curves are not included.   

## Next Steps
- In a future version, we could:

   1. Implement hyperparameter tuning for better model performance.
   2. Explore feature importance to understand the contribution of each variable.
   3. Generate advanced visualizations such as ROC and precision-recall curves.   


# Breast Cancer Diagnosis Using Machine Learning

This Jupyter Notebook demonstrates the application of various machine learning models to predict breast cancer diagnosis based on the Breast Cancer Wisconsin dataset. The main goal is to identify the best-performing model for diagnosing breast cancer.

## Dataset

The dataset used in this project is the [Breast Cancer Wisconsin dataset](https://raw.githubusercontent.com/1SJulioS1/Machine_learning/main/Datasets/Datasets/Breast%20cancer%20Wisconsin/breast-cancer.csv). It contains various features extracted from breast cancer cell images, and the task is to predict whether a tumor is benign (B) or malignant (M) based on these features.

## Notebook Content

The notebook is organized into the following sections:

1. **Data Loading and Preprocessing:** Loading the dataset, encoding the target variable, and splitting the data into training and testing sets.
2. **Support Vector Machine (SVM):** Utilizing the Support Vector Machine algorithm for classification. Hyperparameters are tuned using GridSearchCV.
3. **Random Forest Classifier:** Employing the Random Forest Classifier for classification. Hyperparameter optimization is performed using GridSearchCV.
4. **Decision Tree Classifier:** Using the Decision Tree Classifier for classification. Hyperparameters are tuned using GridSearchCV.
5. **Multi-Layer Perceptron (MLP) Classifier:** Applying the MLP Classifier for classification. Hyperparameters are set and the model is evaluated.
6. **Conclusion:** Concluding remarks on the model performances and the selection of the best model.

## Usage

1. Open the Jupyter Notebook (`breast_cancer_diagnosis.ipynb`) using Jupyter Notebook or Jupyter Lab.
2. Run each code cell sequentially to see the implementation of each model and their evaluation.

## Conclusion

Based on the evaluation of various machine learning models, the Multi-Layer Perceptron (MLP) classifier demonstrates the best performance for predicting breast cancer diagnosis. Its ability to capture complex relationships within the data is likely contributing to its superior accuracy.

## Future Work

- Experiment with different hyperparameter settings for each model to potentially improve their performance.
- Explore additional feature engineering techniques to enhance model accuracy.

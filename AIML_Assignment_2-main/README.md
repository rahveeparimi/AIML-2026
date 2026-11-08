# ML Assignment 2 – Heart Disease Classification

## a. Problem Statement

Predict the presence of heart disease (binary classification) from clinical and medical attributes using multiple machine learning classification models. Build and deploy an interactive Streamlit application that allows users to upload test data, select a model, and view evaluation metrics and confusion matrix.

## b. Dataset Description

UCI Heart Disease dataset (binary classification after converting multi-class `num` target).

- Minimum requirements met: >12 features, >500 instances.
- Key features: age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol (chol), fasting blood sugar (fbs), resting ECG (restecg), maximum heart rate (thalch), exercise-induced angina (exang), oldpeak, slope, number of major vessels (ca), thalassemia (thal).
- Target: `num` converted to binary (0 = no disease, >0 = disease).
- Source: Public UCI repository.

## c. GitHub Repository Link

https://github.com/rahveeparimi/AIML-2026/tree/main/AIML_Assignment_2-main

Contains: `app.py`, `requirements.txt`, `README.md`, `data/Heart_patietns.csv`, model notebook/source.

## d. Models Used

The following 5 classification models were implemented and evaluated on the same dataset (train/test split, stratified):

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Comparison Table

| ML Model Name          | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|------------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression    | 0.810    | 0.902 | 0.819     | 0.843  | 0.831 | 0.614 |
| Decision Tree          | 0.777    | 0.770 | 0.780     | 0.833  | 0.806 | 0.547 |
| KNN                    | 0.826    | 0.886 | 0.872     | 0.804  | 0.837 | 0.654 |
| Naive Bayes            | 0.810    | 0.878 | 0.825     | 0.833  | 0.829 | 0.615 |
| Support Vector Machine | 0.815    | 0.890 | 0.830     | 0.820  | 0.825 | 0.620 |

### Observations

| ML Model Name          | Observation about model performance                                      |
|------------------------|--------------------------------------------------------------------------|
| Logistic Regression    | Strong balanced performance; good AUC and MCC; interpretable.           |
| Decision Tree          | Lower accuracy and AUC; prone to overfitting on this dataset.           |
| KNN                    | Highest accuracy among the set; sensitive to feature scaling.           |
| Naive Bayes            | Competitive results with fast training; assumes feature independence.   |
| Support Vector Machine | Solid performance with good separation; effective on this feature set.  |
| Overall, Winner        | KNN / SVM / Logistic Regression (depending on metric priority)          |

## Streamlit App Features

- CSV test data upload
- Model selection dropdown
- Display of Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix visualization

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Folder Structure

```
AIML_Assignment_2-main/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── Heart_patietns.csv
└── model/
    └── AIML_Assignment_Notebook.ipynb
```

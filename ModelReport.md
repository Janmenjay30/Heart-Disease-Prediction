# In-Depth Model Report: Heart Disease Prediction

**Author:** Janmenjay30
**Date:** November 17, 2025
**Project Repository:** [Heart-Disease-Prediction](https://github.com/Janmenjay30/Heart-Disease-Prediction)

---

## 1. Executive Summary

This report details the end-to-end development of a machine learning model to predict the presence of heart disease based on clinical data from the UCI Cleveland dataset. The primary objective was to build a reliable classification model, prioritizing high **recall** to minimize the risk of false negatives (failing to identify a patient with disease).

- **Four models were trained and evaluated:** Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), and Random Forest.
- **The best-performing model was Logistic Regression**, which achieved a **recall of 92.9%** and an **accuracy of 88.5%** on the unseen test set. It provided an excellent balance of high sensitivity and overall accuracy.
- A complete data processing pipeline was constructed to handle missing values, encode categorical features, and scale numerical data, ensuring reproducibility.
- The final model was packaged into a `Pipeline` object and deployed in a user-friendly **Streamlit application** for interactive, real-time predictions.

---

## 2. Introduction & Problem Statement

Heart disease is a leading cause of death globally. Early and accurate diagnosis is critical for effective treatment and management. Machine learning offers a promising avenue for developing predictive tools that can assist clinicians in identifying at-risk individuals based on standard clinical measurements.

The goal of this project was to answer the question: **"Can we accurately predict the presence of heart disease in a patient using a standard set of clinical features?"**

The project utilized the well-known "processed.cleveland.data" dataset from the UCI Machine Learning Repository, which contains 303 patient records and 13 clinical attributes.

---

## 3. Exploratory Data Analysis (EDA)

A thorough EDA was conducted to understand the dataset's structure, feature distributions, and relationships.

- **Target Variable (`target`):** The dataset was nearly balanced, with approximately 54% of patients having no disease (0) and 46% having some form of heart disease (1). This near-balance meant that accuracy is a reasonable metric, though recall remains the clinical priority.
- **Numerical Features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`):**
  - The `age` distribution was approximately normal, centered around the mid-50s.
  - Patients with heart disease tended to have a lower maximum heart rate (`thalach`) and a higher ST depression value (`oldpeak`).
- **Categorical Features (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`):**
  - **Chest Pain Type (`cp`):** Non-anginal chest pain (type 2) was more common in patients with heart disease, whereas asymptomatic (type 0) was more common in those without.
  - **Thalassemia (`thal`):** A `normal` thalium stress test result was more common in healthy patients, while `fixed defect` was more prevalent in those with disease.
- **Correlations:** A heatmap revealed that `cp`, `thalach`, and `slope` had the strongest positive correlations with the target variable, while `exang`, `oldpeak`, `ca`, and `sex` had the strongest negative correlations.

---

## 4. Data Preprocessing & Feature Engineering

The raw data was transformed into a clean, model-ready format using a reproducible `scikit-learn` pipeline.

1.  **Handling Missing Values:** The dataset used `?` to denote missing values. These were loaded as `np.nan`.
    - For numerical columns (`trestbps`, `chol`, etc.), missing values were imputed using the **median** of the respective column.
    - For categorical columns (`ca`, `thal`), missing values were imputed using the **mode** (most frequent value).
2.  **Categorical Feature Encoding:**
    - Nominal categorical features (`cp`, `restecg`, `slope`, `ca`, `thal`) were converted into numerical format using **One-Hot Encoding**. This creates binary columns for each category, preventing the model from assuming an ordinal relationship.
3.  **Numerical Feature Scaling:**
    - All numerical features were scaled using `StandardScaler`, which standardizes features by removing the mean and scaling to unit variance. This is crucial for distance-based algorithms like KNN and kernel-based algorithms like SVC.
4.  **Pipeline Construction:** All preprocessing steps were encapsulated within a `ColumnTransformer`, which was then placed at the beginning of a `Pipeline` object. This ensures that the exact same transformations are applied to the training, testing, and any new data for prediction, preventing data leakage.

---

## 5. Modeling & Evaluation Strategy

- **Model Selection:** A diverse set of four common classification algorithms was chosen to establish a robust baseline and explore different decision boundaries.
  - **Logistic Regression:** A simple, interpretable linear model.
  - **K-Nearest Neighbors:** A non-parametric, instance-based model.
  - **Support Vector Classifier:** A powerful model that can find complex, non-linear decision boundaries.
  - **Random Forest:** An ensemble of decision trees, robust to overfitting and effective on tabular data.
- **Train-Test Split:** The data was split into an 80% training set and a 20% testing set. The split was **stratified** by the `target` variable to ensure that the proportion of patients with and without heart disease was the same in both the train and test sets.
- **Evaluation Metrics:** The primary metric for model selection was **Recall** for the positive class (disease present). Other standard metrics were also recorded:
  - **Accuracy:** Overall correct predictions.
  - **Precision:** Of the patients predicted to have disease, how many actually did.
  - **F1-Score:** The harmonic mean of Precision and Recall.
  - **Confusion Matrix:** To visualize the counts of true/false positives and negatives.

---

## 6. Results & Model Comparison

All models were trained on the same preprocessed training data and evaluated on the unseen test set. The results are summarized below:

| Model                | Accuracy | Precision | Recall | F1-Score | False Negatives (FN) |
| -------------------- | -------- | --------- | ------ | -------- | -------------------- |
| **Logistic Regression** | **0.885**  | **0.839**   | **0.929** | **0.881**  | **2**                  |
| **SVC**                | 0.885    | 0.839     | 0.929  | 0.881    | 2                    |
| **KNN (k=5)**          | 0.885    | 0.862     | 0.893  | 0.877    | 3                    |
| **Random Forest**      | 0.869    | 0.813     | 0.929  | 0.867    | 2                    |

**Key Observations:**
- All models performed exceptionally well, with accuracies nearing 90%.
- **Logistic Regression, SVC, and Random Forest all achieved the highest possible recall of 92.9%**, correctly identifying 26 out of 28 patients with heart disease in the test set and missing only 2 (False Negatives).
- KNN had slightly lower recall but the highest precision, meaning it was the most "confident" when it did predict disease, resulting in fewer false alarms (False Positives).

---

## 7. Final Model Selection

**The final selected model is the Logistic Regression pipeline.**

**Justification:**
- **Top-Tier Recall:** It achieved the highest recall score (92.9%), tying with SVC and Random Forest, thus meeting the primary clinical objective of minimizing missed cases.
- **Simplicity and Interpretability:** As a linear model, Logistic Regression is far more interpretable than SVC or Random Forest. Its coefficients could be inspected to understand the influence of each feature on the prediction, which is a significant advantage in a clinical context.
- **Excellent Performance:** It delivered the same top-tier accuracy and F1-score as the more complex SVC model, demonstrating that a simpler model was sufficient for this problem.

The entire pipeline, including the preprocessor and the trained Logistic Regression model, was saved to `models/best_pipeline.joblib`.

---

## 8. Deployed Application

A simple web application was built using **Streamlit** to provide an interactive interface for the trained model.

- **File:** `app_streamlit.py`
- **Functionality:** The application presents a form where a user can input the 13 clinical features. Upon submission, the app loads the saved `best_pipeline.joblib` object, applies the full preprocessing pipeline to the input data, and predicts the probability of heart disease.
- **To Run:**
  ```bash
  streamlit run app_streamlit.py
  ```

This application serves as a proof-of-concept for how the model could be integrated into a clinical workflow.

---

## 9. Conclusion & Future Work

This project successfully developed and evaluated a machine learning pipeline for heart disease prediction, culminating in a Logistic Regression model that is both highly sensitive (92.9% recall) and accurate (88.5%). The project demonstrates the feasibility of using standard clinical data to build effective diagnostic aids.

**Future Work:**
- **Hyperparameter Tuning:** Although the baseline models performed well, `GridSearchCV` could be used to tune the hyperparameters of the Random Forest or SVC models to potentially increase precision without sacrificing recall.
- **Feature Importance Analysis:** A deeper dive into the feature importances from the Random Forest model or the coefficients from the Logistic Regression model would provide more clinical insights into the key drivers of prediction.
- **External Validation:** The model should be tested on other heart disease datasets (e.g., from different hospitals or countries) to assess its generalizability.
- **Advanced Modeling:** Explore more advanced techniques like Gradient Boosting (XGBoost, LightGBM) or neural networks to see if performance can be further improved.
- **App Enhancement:** The Streamlit app could be enhanced with data validation, explanations for the prediction (e.g., using SHAP), and a more polished user interface.

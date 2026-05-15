# Model Improvement Log

This file tracks the enhancements made to our models, focusing particularly on handling imbalanced datasets (like the Framingham dataset).

## Baseline Metrics (Class Weights)

Before applying SMOTE, the Framingham dataset handled its 18% minority class imbalance using the `class_weight="balanced"` parameter in the models. 

**Best Model**: SVC
**Baseline Performance:**
- **Accuracy**: 68.40%
- **Precision**: 25.44%
- **Recall**: 55.81%
- **F1 Score**: 34.95%
- **ROC-AUC**: 68.32%

*Note: While Logistic Regression achieved a slightly higher recall (58.14%), SVC was selected as the best model due to its slightly higher F1 score.*

---

## Enhancement: Applying SMOTE

We replaced simple class weighting with **SMOTE** (Synthetic Minority Over-sampling Technique) using the `imbalanced-learn` library. SMOTE creates synthetic examples of the minority class to balance the training data, which often leads to better decision boundaries for models that struggle with severe imbalance.

### Post-SMOTE Performance:
**Best Model**: Logistic Regression
- **Accuracy**: 65.80% *(Slight drop from 68.40%)*
- **Precision**: 24.28%
- **Recall**: 58.91% *(Improved from 55.81% on previous best model)*
- **F1 Score**: 34.39%
- **ROC-AUC**: 69.67% *(Improved from 68.32%)*

### Key Insights:
1. **Recall Improvement**: The most important clinical metric (Recall) improved to **58.91%** using Logistic Regression with SMOTE.
2. **Massive gains for Tree/Distance Models**: 
   - **Random Forest Recall** jumped from an abysmal **3.88%** to **20.93%**.
   - **XGBoost Recall** jumped from **9.30%** to **27.91%**.
   - **KNN Recall** surged from **7.75%** to **48.84%**.
3. **Trade-off**: As expected when aggressively oversampling the minority class to maximize recall, there was a slight drop in overall accuracy (due to an increase in false positives). However, in a medical screening context, catching more true positive cases (higher recall) is worth the trade-off.

---

## Enhancement: Feature Selection (Dropping Weak Features)

Based on a correlation analysis with the target `TenYearCHD`, several features in the Framingham dataset were found to be very noisy with extremely weak correlations (< 0.08). We removed the following features: `currentSmoker`, `heartRate`, `cigsPerDay`, `prevalentStroke`, and `BMI`.

### Post-Feature Selection Performance (with SMOTE):
**Best Model**: SVC
- **Accuracy**: 68.87% *(Improved from 65.80%)*
- **Precision**: 24.72%
- **Recall**: 51.16% 
- **F1 Score**: 33.33%
- **ROC-AUC**: 67.56%

*(Note: Logistic Regression maintained a higher recall of 58.91%, but SVC was selected due to a slightly better balance in the F1 score).*

### Key Insights:
1. **Accuracy Recovery**: By removing noisy features, the overall accuracy of the best model recovered significantly (from 65.8% back up to 68.87%).
2. **Random Forest Jump**: The Random Forest model benefited the most from removing noisy features. Its recall increased further from 20.93% up to **26.36%**.
3. **Cleaner Signals**: Tree-based models (Random Forest, XGBoost) perform noticeably better when unhelpful features don't cloud the splits.

---

## Enhancement: Tuning Prediction Threshold

By default, classification models use a decision threshold of `0.5` (50% probability). However, in imbalanced medical datasets where missing a disease is far worse than a false alarm, we can lower this threshold to increase **Recall**. 

We modified the evaluation pipeline to utilize `predict_proba()` and lowered the prediction threshold from `0.5` to `0.35`.

### Post-Threshold Tuning Performance (Threshold = 0.35):
**Best Model**: SVC
- **Accuracy**: 57.55% *(Drop from 68.87%)*
- **Precision**: 22.70%
- **Recall**: 74.42% *(Massive improvement from 51.16%)*
- **F1 Score**: 34.78%
- **ROC-AUC**: 67.56%

*(Note: Logistic Regression achieved an incredible **84.50% Recall**, but SVC was ultimately selected by the pipeline for maintaining a slightly better balance in the overall F1 score).*

### Key Insights:
1. **Unprecedented Recall**: Lowering the threshold to `0.35` allowed Logistic Regression to catch **84.5%** of all true positive cases, and SVC caught **74.4%**. This is a dramatic improvement over the original ~50% baseline.
2. **The Trade-off Cost**: As the threshold is lowered, the model becomes highly sensitive. This naturally results in a steep drop in accuracy (down to ~57%) because it is predicting "Disease" much more aggressively, causing many more false positives.
3. **Conclusion**: Tuning the threshold is the most effective way to explicitly force the model to prioritize Recall over Accuracy, fulfilling the primary clinical objective of the project.

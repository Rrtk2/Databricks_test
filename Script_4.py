#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, 
                             roc_auc_score, confusion_matrix, roc_curve)
import joblib #%pip install joblib


# -------------------------
# Imports
# -------------------------
synthetic_df = joblib.load('synthetic_df.pkl')
X = joblib.load('X.pkl')
y = joblib.load('y.pkl')




# -------------------------
# Step 8 (Improved): Train a classifier using Repeated Stratified K-Fold cross validation 
# and then use the trained model to predict synthetic samples.
#
# In this step, we:
#  - Use RepeatedStratifiedKFold to ensure that each fold preserves the percentage of samples for each class,
#    and we repeat the split multiple times (5 splits, 10 repeats) for robust performance estimation.
#  - Evaluate the classifier (Logistic Regression) using multiple scoring metrics:
#      - Accuracy: The proportion of correctly classified samples.
#      - Macro F1 Score: The harmonic mean of precision and recall, computed per class and averaged.
#      - ROC AUC (OvR): The area under the ROC curve for the one-vs-rest setting.
#  - Print the cross-validation performance (mean and standard deviation) for training and test splits.
#  - Finally, fit the classifier on the full original dataset and predict on the synthetic dataset.
# -------------------------

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

# Define the classifier with Logistic Regression for multiclass classification.
# Note: 'lbfgs' is used as the solver here and max_iter is increased to 200 for convergence.
clf = LogisticRegression(max_iter=200, multi_class='multinomial', solver='lbfgs')

# Set up repeated stratified k-fold cross validation parameters:
#   - 5 splits (folds) per repetition
#   - 10 repetitions
#   - A fixed random state for reproducibility.
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

# Define scoring metrics for cross validation.
scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',
    'roc_auc_ovr': 'roc_auc_ovr'
}

# Perform cross validation on the original iris dataset (X, y)
cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=True)

# Print the mean and standard deviation for training and test metrics.
print("Cross-validation results on the original Iris dataset:")
print(f"Mean Training Accuracy: {np.mean(cv_results['train_accuracy']):.3f} ± {np.std(cv_results['train_accuracy']):.3f}")
print(f"Mean Test Accuracy: {np.mean(cv_results['test_accuracy']):.3f} ± {np.std(cv_results['test_accuracy']):.3f}")
print(f"Mean Test Macro F1 Score: {np.mean(cv_results['test_f1_macro']):.3f} ± {np.std(cv_results['test_f1_macro']):.3f}")
print(f"Mean Test ROC AUC (OvR): {np.mean(cv_results['test_roc_auc_ovr']):.3f} ± {np.std(cv_results['test_roc_auc_ovr']):.3f}")

# After evaluating via repeated cross validation, fit the classifier on the full original dataset.
clf.fit(X, y)

# Prepare the synthetic dataset for prediction.
# We assume that 'synthetic_df' contains the same feature columns as X and a 'target' column.
X_synthetic = synthetic_df.drop(columns=['target'])
y_synthetic = synthetic_df['target']

# Make predictions on the synthetic data.
y_pred = clf.predict(X_synthetic)
y_prob = clf.predict_proba(X_synthetic)

print("\nCompleted predictions on synthetic samples using a classifier trained with repeated cross-validation.")


# -------------------------
# Step 9 (Updated for Excel Output using xlsxwriter):
# Calculate, store, and display all classification metrics in an Excel workbook.
#
# In this step, we:
#   - Compute overall metrics: accuracy, Matthews correlation coefficient, and macro F1 score.
#   - Compute the confusion matrix.
#   - Derive per-class sensitivity (recall) and specificity.
#   - Organize each set of metrics in a pandas DataFrame.
#   - Save the results to an Excel workbook with separate sheets for easy interpretation,
#     using 'xlsxwriter' as the engine instead of openpyxl.
# -------------------------

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

# Compute overall metrics for the synthetic dataset.
overall_metrics = {
    'Accuracy': accuracy_score(y_synthetic, y_pred),
    'Matthews Corrcoef': matthews_corrcoef(y_synthetic, y_pred),
    'Macro F1 Score': f1_score(y_synthetic, y_pred, average='macro')
}

# Create a DataFrame for overall metrics.
df_overall = pd.DataFrame(overall_metrics, index=[0])
print("\nOverall Classification Metrics on Synthetic Data:")
print(df_overall)

# Compute the confusion matrix.
cm = confusion_matrix(y_synthetic, y_pred)
# Create a DataFrame for the confusion matrix with appropriate labels.
class_labels = list(clf.classes_)
df_confusion = pd.DataFrame(cm, index=[f"Actual_{cls}" for cls in class_labels],
                                 columns=[f"Predicted_{cls}" for cls in class_labels])
print("\nConfusion Matrix:")
print(df_confusion)

# Compute per-class sensitivity (recall) and specificity.
per_class_data = []
for i, cls in enumerate(class_labels):
    TP = cm[i, i]
    FN = np.sum(cm[i, :]) - TP
    FP = np.sum(cm[:, i]) - TP
    TN = np.sum(cm) - (TP + FN + FP)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    per_class_data.append({
        'Class': cls,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    })
    print(f"Class {cls}: Sensitivity (Recall) = {sensitivity:.3f}, Specificity = {specificity:.3f}")

# Create a DataFrame for per-class metrics.
df_per_class = pd.DataFrame(per_class_data)

# Save all metrics to an Excel file using xlsxwriter instead of openpyxl.
excel_path = 'classification_metrics.xlsx'
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    df_overall.to_excel(writer, sheet_name='Overall Metrics', index=False)
    df_confusion.to_excel(writer, sheet_name='Confusion Matrix')
    df_per_class.to_excel(writer, sheet_name='Per-Class Metrics', index=False)

print(f"\nSaved all classification metrics to {excel_path}")



# -------------------------
# Step 10: Compute ROC curves and AUC for each class (one-vs-rest)
# -------------------------
fpr = {}
tpr = {}
roc_auc = {}

plt.figure()
for i, cls in enumerate(clf.classes_):
    # Binary labels: current class vs rest
    binary_true = (y_synthetic == cls).astype(int)
    binary_prob = y_prob[:, i]
    fpr[cls], tpr[cls], _ = roc_curve(binary_true, binary_prob)
    roc_auc[cls] = roc_auc_score(binary_true, binary_prob)
    plt.plot(fpr[cls], tpr[cls], label=f'Class {cls} (AUC = {roc_auc[cls]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves on Synthetic Data')
plt.legend(loc='best')
roc_plot_path = 'roc_curve.png'
plt.savefig(roc_plot_path)
plt.close()
print(f"Saved ROC curve plot to {roc_plot_path}")

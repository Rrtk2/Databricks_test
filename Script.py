#!/usr/bin/env python
# -------------------------
# Installer
# -------------------------
import subprocess, sys, os, argparse

def install_requirements(req_file):
    if not os.path.isfile(req_file):
        print(f"Requirements file not found: {req_file}", file=sys.stderr)
        return False
    command = [sys.executable, '-m', 'pip', 'install', '-r', req_file]
    print("Running:", " ".join(command))
    try:
        subprocess.check_call(command)
        return True
    except subprocess.CalledProcessError as e:
        print("Installation error:", e, file=sys.stderr)
        return False
    
parser = argparse.ArgumentParser(description="Install packages from a requirements file.")
parser.add_argument("-f", "--file", default="requirements.txt", help="Path to requirements file")
args = parser.parse_args()

if not install_requirements(args.file):
    sys.exit(1)
print("Installation completed successfully.")    

    
# -------------------------
# Script start
# -------------------------    
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib #%pip install joblib

# For PCA and classification
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, 
                             roc_auc_score, confusion_matrix, roc_curve)

import statsmodels.api as sm

# -------------------------
# Step 1: Load iris dataset and save to CSV
# -------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save the original dataset to CSV
csv_path = 'iris.csv'
df.to_csv(csv_path, index=False)
print(f"Saved iris dataset to {csv_path}")

# -------------------------
# Step 2: Load the CSV
# -------------------------
df_loaded = pd.read_csv(csv_path)
X = df_loaded.drop(columns=['target'])
y = df_loaded['target']

# -------------------------
# Step 3: Perform PCA on the full feature set and plot
# -------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X)

plt.figure()
sc = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title('PCA of Iris Dataset (All Features)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(sc, label='Target')
pca_plot_path = 'pca_plot.png'
plt.savefig(pca_plot_path)
plt.close()
print(f"Saved PCA plot to {pca_plot_path}")

# -------------------------
# Step 4: Linear Modeling to extract significant features
# Using a logistic regression (one-vs-rest) via statsmodels for each class
# -------------------------
significant_features = set()
for cls in sorted(y.unique()):
    # Create a binary outcome: 1 if sample is of the current class, else 0
    y_binary = (y == cls).astype(int)
    # Add constant for intercept
    X_const = sm.add_constant(X)
    model = sm.Logit(y_binary, X_const)
    result = model.fit(disp=False)
    # Extract features with p-value less than 0.05 (exclude constant)
    sig = result.pvalues[result.pvalues < 0.05].index
    sig = [feat for feat in sig if feat != 'const']
    significant_features.update(sig)
    print(f"Class {cls}: Significant features found: {sig}")

if len(significant_features) == 0:
    sig_features = X.columns.tolist()
    print("No features met the significance threshold; using all features.")
else:
    sig_features = list(significant_features)
    print(f"Union of significant features across classes: {sig_features}")

# -------------------------
# Step 5: PCA on significant features and plot
# -------------------------
pca_sig = PCA(n_components=2)
pca_sig_data = pca_sig.fit_transform(X[sig_features])

plt.figure()
sc2 = plt.scatter(pca_sig_data[:, 0], pca_sig_data[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title('PCA on Significant Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(sc2, label='Target')
pca_sig_plot_path = 'pca_sig_plot.png'
plt.savefig(pca_sig_plot_path)
plt.close()
print(f"Saved PCA plot (significant features) to {pca_sig_plot_path}")

# -------------------------
# Step 6: Save the PCA object (from the full-feature PCA)
# -------------------------
pca_object_path = 'pca_object.pkl'
joblib.dump(pca, pca_object_path)
print(f"Saved PCA object to {pca_object_path}")

# -------------------------
# Step 7 (Improved): Generate synthetic samples with a multivariate approach
# -------------------------

num_samples_per_class = 100
synthetic_samples = []

# We’ll sample from the empirical mean and covariance for each class in the *entire* feature set.
# If you still want to limit to significant features only, see the note below.
for cls in sorted(y.unique()):
    # Subset of data belonging to the current class
    class_data = X[y == cls]

    # Estimate the mean vector (length = number of features)
    mean_vector = class_data.mean().values
    
    # Estimate the covariance matrix (shape: features x features)
    cov_matrix = class_data.cov().values
    
    # Sample from the multivariate normal distribution
    # If the dataset has singular covariance (rare in Iris, more common with many features),
    # you may need regularization or a different approach.
    synth_class = np.random.multivariate_normal(
        mean=mean_vector, 
        cov=cov_matrix, 
        size=num_samples_per_class
    )
    
    # Create a DataFrame for the synthetic data
    synth_class_df = pd.DataFrame(synth_class, columns=X.columns)
    synth_class_df['target'] = cls
    
    synthetic_samples.append(synth_class_df)

synthetic_df = pd.concat(synthetic_samples, ignore_index=True)

# Optionally ensure the synthetic features stay within some reasonable bounds
# (e.g., clamp negative values if your domain knowledge says features can't be negative).

# Save synthetic data
synthetic_csv_path = 'synthetic_iris.csv'
synthetic_df.to_csv(synthetic_csv_path, index=False)
print(f"Saved improved synthetic samples to {synthetic_csv_path}")



# -------------------------
# Step X: Perform PCA on the synth data
# -------------------------
pca = PCA(n_components=2)
pca_synth_data = pca.fit_transform(synthetic_df)
y_synth = synthetic_df.iloc[:,4]
plt.figure()
sc = plt.scatter(pca_synth_data[:, 0], pca_synth_data[:, 1], c=y_synth, cmap='viridis', edgecolor='k')
plt.title('PCA of Iris Dataset (All Features)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(sc, label='Target')
pca_plot_path = 'pca_plot_syth.png'
plt.savefig(pca_plot_path)
plt.close()
print(f"Saved PCA plot to {pca_plot_path}")


# -------------------------
# Step X: Combined PCA plot (original vs synthetic, different markers)
# Use the original PCA (saved from full-feature PCA) to project synthetic samples.
# -------------------------
import joblib

# Load the original PCA object
pca_original = joblib.load('pca_object.pkl')

# Ensure we are using the same set of features as was used originally.
features = iris.feature_names  # same order as in original X

# Project the original data using the original PCA
X_proj = pca_original.transform(X[features])

# Project the synthetic samples using the same PCA object
X_synth_proj = pca_original.transform(synthetic_df[features])

# Create a combined PCA plot
plt.figure(figsize=(8, 6))

# Plot the original iris data
sc_orig = plt.scatter(
    X_proj[:, 0],
    X_proj[:, 1],
    c=y,
    cmap='viridis',
    edgecolor='k',
    marker='o',
    vmin=0,
    vmax=2,
    label='Original'
)

# Plot the synthetic data with different marker
sc_synth = plt.scatter(
    X_synth_proj[:, 0],
    X_synth_proj[:, 1],
    c=synthetic_df['target'],
    cmap='viridis',
    edgecolor='k',
    marker='x',
    vmin=0,
    vmax=2,
    label='Synthetic'
)

plt.title('Combined PCA Plot: Original vs Synthetic Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Add a shared colorbar (based on the original scatter, they share same mapping)
plt.colorbar(sc_orig, label='Target')

# Add a legend for the markers
plt.legend()

combined_pca_plot_path = 'combined_pca_plot.png'
plt.savefig(combined_pca_plot_path)
plt.close()
print(f"Saved combined PCA plot to {combined_pca_plot_path}")



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

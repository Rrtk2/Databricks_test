#!/usr/bin/env python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib #%pip install joblib
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, 
                             roc_auc_score, confusion_matrix, roc_curve)


# -------------------------
# argparse
# -------------------------
import argparse    
parser = argparse.ArgumentParser()
parser.add_argument("-num_of_samples", type=int, default=100, help="Number of synthetic samples to generate per class")
args = parser.parse_args()

# -------------------------
# Imports
# -------------------------
X = joblib.load('X.pkl')
y = joblib.load('y.pkl')
iris = joblib.load('iris.pkl')





# -------------------------
# Step 7 (Improved): Generate synthetic samples with a multivariate approach
# -------------------------

num_samples_per_class = args.num_of_samples
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
# Exports
# -------------------------
joblib.dump(synthetic_df,'synthetic_df.pkl')
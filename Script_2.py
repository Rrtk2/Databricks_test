#!/usr/bin/env python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import joblib #%pip install joblib

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
# Exports
# -------------------------
joblib.dump(X,'X.pkl')
joblib.dump(y, 'y.pkl')
joblib.dump(iris, 'iris.pkl')

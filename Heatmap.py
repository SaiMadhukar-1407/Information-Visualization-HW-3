import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pyod.models.cblof import CBLOF
from pyod.models.lof import LOF
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.pca import PCA as PCA_OD
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.feature_bagging import FeatureBagging

import warnings
warnings.filterwarnings("ignore")

# === Dataset and Algorithm Setup ===
datasets = [
    ("Dataset 1 Wine Quality", "plasma"),    # Wine Quality
    ("Dataset 2 Bank Marketing", "cividis"),   # Bank Marketing
    ("Dataset 3 Online Retail", "turbo"),     # Online Retail
]

model_categories = {
    "Proximity-Based": [("CBLOF", CBLOF()), ("LOF", LOF())],
    "Probabilistic": [("GMM", GMM()), ("KDE", KDE())],
    "Linear Model": [("PCA", PCA_OD()), ("OCSVM", OCSVM())],
    "Ensemble": [("IForest", IForest()), ("FeatureBagging", FeatureBagging())],
}

# === Load Preprocessed PCA-Reduced CSV ===
def load_preprocessed_csv(name):
    df = pd.read_csv(f"preprocessed_{name.replace(' ', '_')}.csv")
    return df[["PC1", "PC2"]].values

# === Main Plotting Function ===
def plot_final_colormap_grid():
    fig, axes = plt.subplots(len(model_categories), len(datasets) * 2, figsize=(18, 11))

    for col_idx, (ds_name, cmap_name) in enumerate(datasets):
        X_2D = load_preprocessed_csv(ds_name)
        x_min, x_max = X_2D[:, 0].min() - 1, X_2D[:, 0].max() + 1
        y_min, y_max = X_2D[:, 1].min() - 1, X_2D[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]

        for row_idx, (cat_name, models) in enumerate(model_categories.items()):
            for i, (model_name, model) in enumerate(models):
                ax = axes[row_idx, col_idx * 2 + i]

                # Reconstruct 2D->original->fit model
                # We'll assume you already saved the full original data PCA -> for now retrain model on PCA space
                model.fit(X_2D)
                zz = model.decision_function(grid).reshape(xx.shape)
                zz_norm = (zz - zz.min()) / (zz.max() - zz.min())

                im = ax.contourf(xx, yy, zz_norm, levels=100, cmap=cmap_name)
                ax.scatter(X_2D[:, 0], X_2D[:, 1], s=2, c='k', alpha=0.6)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(model_name, fontsize=8)

                # Category name
                if col_idx == 0 and i == 0:
                    ax.text(-0.3, 0.5, cat_name, fontsize=10, weight='bold',
                            va='center', ha='right', transform=ax.transAxes, rotation=90)

                # Dataset name
                if row_idx == 0 and i == 0:
                    ax.text(0.5, 1.15, ds_name, fontsize=10, weight='bold',
                            ha='center', transform=ax.transAxes)

        # Add colorbar
        cbar_ax = fig.add_axes([0.93, 0.15 + 0.2 * (2 - col_idx), 0.015, 0.15])
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.set_yticks([0.0, 1.0])
        cbar.ax.set_yticklabels(["Likely Inlier", "Likely Outlier"])
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_ylabel("Outlier Likelihood", fontsize=9)

    plt.suptitle("Outlier Detection Heatmap Comparison (Grouped by Algorithm Type)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig("final_outlier_colormap_grid.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_final_colormap_grid()

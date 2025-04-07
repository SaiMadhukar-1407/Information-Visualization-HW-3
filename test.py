# === Modified Code for Paper-Style Grid Format (Dataset-wise Columns with Unified Category Rows) ===
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
    ("Dataset 1 Wine Quality", "plasma"),
    ("Dataset 2 Bank Marketing", "cividis"),
    ("Dataset 3 Online Retail", "turbo")
]

model_categories = {
    "Proximity-Based": [("CBLOF", CBLOF()), ("LOF", LOF())],
    "Probabilistic": [("GMM", GMM()), ("KDE", KDE())],
    "Linear Model": [("PCA", PCA_OD()), ("OCSVM", OCSVM())],
    "Ensemble": [("IForest", IForest()), ("FeatureBagging", FeatureBagging())],
}

# === Load Preprocessed CSV ===
def load_preprocessed_csv(name):
    formatted_name = name.replace(" ", "_")
    df = pd.read_csv(f"preprocessed_{formatted_name}.csv")
    return df[["PC1", "PC2"]].values

# === Final Grid Plotting ===
def plot_column_aligned_heatmap():
    fig, axes = plt.subplots(nrows=len(model_categories)*2, ncols=len(datasets), figsize=(13, 12))

    for col_idx, (ds_label, cmap_name) in enumerate(datasets):
        X_2D = load_preprocessed_csv(ds_label)
        x_min, x_max = X_2D[:, 0].min() - 1, X_2D[:, 0].max() + 1
        y_min, y_max = X_2D[:, 1].min() - 1, X_2D[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]

        row_idx = 0
        for cat_name, models in model_categories.items():
            for model_name, model in models:
                ax = axes[row_idx, col_idx]
                model.fit(X_2D)
                zz = model.decision_function(grid).reshape(xx.shape)
                zz_norm = (zz - zz.min()) / (zz.max() - zz.min())

                im = ax.contourf(xx, yy, zz_norm, levels=100, cmap=cmap_name)
                ax.scatter(X_2D[:, 0], X_2D[:, 1], s=2, c='k', alpha=0.6)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(model_name, fontsize=8)

                # Left-most column category label
                if col_idx == 0 and model_name == models[0][0]:
                    ax.text(-0.28, 0.5, cat_name, fontsize=10, weight='bold',
                            va='center', ha='right', transform=ax.transAxes, rotation=90)

                # Top column dataset title
                if row_idx == 0:
                    ax.set_title(f"{ds_label}\n{model_name}", fontsize=9, weight='bold')
                row_idx += 1

        # Add colorbar
        cbar_ax = fig.add_axes([0.30 + 0.22 * col_idx, 0.1, 0.01, 0.75])
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(["Likely Inlier", "Likely Outlier"])
        cbar.ax.set_ylabel("Outlier Likelihood", fontsize=9)

    plt.suptitle("Outlier Detection Heatmap Comparison (Paper-Style Layout)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("paper_style_grid_layout.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_column_aligned_heatmap()
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo

datasets = [
    ("Dataset 1 Wine Quality", 186),  # Wine Quality
    ("Dataset 2 Bank Marketing", 222),  # Bank Marketing
    ("Dataset 3 Online Retail", 352),  # Online Retail
]

def preprocess_and_save(dataset_id, dataset_name):
    data = fetch_ucirepo(id=dataset_id)
    X = data.data.features.select_dtypes(include=[np.number])

    if dataset_id == 352:
        X = X.dropna(axis=1, thresh=int(0.95 * len(X)))
        X = X.dropna().sample(n=500, random_state=42)
    else:
        X = X.dropna()

    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)

    df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df.to_csv(f"preprocessed_{dataset_name.replace(' ', '_')}.csv", index=False)

for name, ds_id in datasets:
    preprocess_and_save(ds_id, name)

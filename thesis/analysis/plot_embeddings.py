from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import zarr

from msc import config

figures_path = r"results/figures"

cache_path = r"data/cache.zarr"

cache_zarr = zarr.open(cache_path, 'r')

ds_zarr = cache_zarr[f"{config['dataset_id']}"]

times = []
embeddings = []
for key in sorted([int(k) for k in ds_zarr.keys() if k != 'mu' and k != 'std']):
    time_zarr = ds_zarr[f'{key}']
    if 'embedding' in time_zarr:
        times.append(key)
        embedding = time_zarr['embedding'][1:9]
        embeddings.append(embedding)

X = np.stack(embeddings)
pca = PCA(n_components=2)

components = pca.fit_transform(X)
hue_temp = np.zeros_like(X[:,0])
hue_temp[:200] = 5
sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=hue_temp)
sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=times).set(title='Embeddings')

plt.savefig(f"{figures_path}/embeddings.png")
plt.savefig(f"{figures_path}/temp.pdf", bbox_inches='tight')
plt.clf()
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import zarr

from msc import config
from thesis.experiment.main import OfflineExperiment

figures_path = r"results/figures"

cache_path = r"data/cache.zarr"

cache_zarr = zarr.open(cache_path, 'r')

ds_zarr = cache_zarr[f"{config['dataset_id']}"]

# collect embeddings from ds_zarr
embeddings = []
exclude_groups = ['std', 'mu', 'events']
samples_df = []
for key in sorted([int(k) for k in ds_zarr.keys() if k not in exclude_groups]):
    time_zarr = ds_zarr[f'{key}']
    if 'embedding' in time_zarr:
        embedding = time_zarr['embedding'][1:9]
        embeddings.append(embedding)
        data = {"time": key,
                "embedding": embedding}
        samples_df.append(data)

# collect embeddings from events_zarr
events_zarr = ds_zarr['events']
for key in sorted([int(k) for k in events_zarr.keys() if k not in exclude_groups]):
    time_zarr = events_zarr[f'{key}']
    if 'embedding' in time_zarr:
        embedding = time_zarr['embedding'][1:9]
        embeddings.append(embedding)
        data = {"time": key,
                "embedding": embedding}
        samples_df.append(data)

# create samples_df DataFrame
samples_df = pd.DataFrame(samples_df)

# compute time to event and add to samples_df
experiment = OfflineExperiment(config)
events = experiment.get_event_sample_times()
events_df = pd.DataFrame(events, columns=['onset'])
events_df = events_df.sort_values(by='onset', ignore_index=True)
samples_df = samples_df.sort_values(by='time', ignore_index=True)
samples_df = pd.merge_asof(samples_df, events_df, left_on='time', right_on='onset', direction='forward')
samples_df['time_to_event'] = samples_df['onset'] - samples_df['time']

# compute PCA and add to samples_df
pca = PCA(n_components=2)
components = pca.fit_transform(np.stack(samples_df['embedding']))  # type: ignore
samples_df[['pca-2d-one', 'pca-2d-two']] = components

# binarize time_to_event
def get_class(x):
        if x < 5:
                return 0
samples_df['class'] = samples_df['time_to_event'].apply(lambda x: x if x <= 5 else 10)

# plot plots
plt.clf()
samples_df['time_to_event'].hist()
plt.title('time to event')
plt.savefig(f"{figures_path}/hist.pdf", bbox_inches='tight')

plt.clf()
plt.style.use(['science', 'no-latex'])
sns.scatterplot(data=samples_df, x='pca-2d-one', y='pca-2d-two', hue='class', palette='crest').set(title='Embeddings', xlabel="PC1", ylabel="PC2")
plt.savefig(f"{figures_path}/embeddings.pdf", bbox_inches='tight')

# plt.clf()
# plt.savefig(f"{figures_path}/temp.pdf", bbox_inches='tight')

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from msc import config
from msc.cache_handler import get_samples_df
from msc.data_utils import get_config_dataset, get_event_sample_times
from msc.plot_utils import set_size


def plot(width):
        figures_path = r"results/figures"
        
        # load samples_df
        samples_df = get_samples_df(config['dataset_id'], with_events=True)  # type: ignore

        # get ds
        ds = get_config_dataset()
        # compute time to event and add to samples_df
        events = get_event_sample_times(ds, augment=False)
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
        # plt.clf()
        # samples_df['time_to_event'].hist()
        # plt.title('time to event')
        # plt.savefig(f"{figures_path}/embeddings/hist.pdf", bbox_inches='tight')

        plt.clf()
        plt.style.use(['science', 'no-latex'])
        # from matplotlib import rcParams
        # rcParams['figure.figsize'] = set_size(width)
        # sns.set(rc={'figure.figsize': set_size(width)})
        # plt.scatter(samples_df['pca-2d-one'], samples_df['pca-2d-two'])
        # plt.xlabel("PC1")
        # plt.ylabel("PC2")
        g = sns.scatterplot(data=samples_df, x='pca-2d-one', y='pca-2d-two', hue='class', palette='crest')
        g.set(xlabel="PC1", ylabel="PC2")
        g.figure.set_size_inches(*set_size(width))
        plt.savefig(f"{figures_path}/embeddings/embeddings.pdf", bbox_inches='tight')


        # use this to get the axis limits (for other plots)
        # xmin, xmax, ymin, ymax = plt.axis()

        # s = 'xmin = ' + str(round(xmin, 2)) + ', ' + \
        #     'xmax = ' + str(xmax) + '\n' + \
        #     'ymin = ' + str(ymin) + ', ' + \
        #     'ymax = ' + str(ymax) + ' '

        # print(f"{s=}")
        # plt.clf()
        # plt.savefig(f"{figures_path}/temp.pdf", bbox_inches='tight')

if __name__ == "__main__":
        plot(width=380.9)
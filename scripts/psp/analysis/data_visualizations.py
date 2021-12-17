import matplotlib.pyplot as plt
import pandas as pd

def plot_pca_projection(dataset_fpath):
    # load the data
    dataset_df = pd.read_csv(dataset_fpath)
    # standardize the data


    # PCA project to 2D

    # plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)

    labels = ['interictal', 'preictal']
    colors = ['g', 'p']
    for target, color in zip(labels, colors):
        indicesToKeep = dataset_df['label_desc'] == target
        ax.scatter(dataset_df.loc[indicesToKeep, 'principal component 1']
                   , dataset_df.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(labels)
    ax.grid()
    set


dataset_dir = r"C:\Users\noam\Repositories\noamsgl\msc\results\epilepsiae\max_cross_corr\surfCO\pat_3500\20211213T182128"

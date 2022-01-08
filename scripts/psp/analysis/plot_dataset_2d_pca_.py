import os
import sys
from tkinter import messagebox
from tkinter.filedialog import askdirectory

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from msc.dataset import PSPDataset


def plot_pca_projection(dataset_df_row, ax=None):
    print(f"plotting pca for {dataset_df_row}")
    # load the data
    dataset = PSPDataset(dataset_df_row.data_dir)
    X, labels, samples_df = dataset.get_X(), dataset.get_labels(), dataset.samples_df

    # standardize the data
    X = StandardScaler().fit_transform(X)

    # PCA project to 2D
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDF = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDF, samples_df.label_desc], axis=1)

    # plot
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title(f"{dataset_df_row.patient_name}, {dataset_df_row.feature_name}")

    labels = ['interictal', 'preictal']
    colors = ['g', 'm']
    for target, color in zip(labels, colors):
        indicesToKeep = finalDf['label_desc'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(labels)
    ax.grid()


if __name__ == '__main__':
    # get the CurrentStudy dataset directory
    if not messagebox.askokcancel(title='PSP',
                                  message="Welcome to the Physiological Signal Processing data visualizer!\n"
                                          "Please select the dataset directory:"):
        sys.exit(-1)

    init_dir = os.getcwd()
    # show an "Open" dialog box and return the path to the selected file
    data_dir = askdirectory(initialdir=init_dir)
    if not data_dir:
        sys.exit(-1)

    # ax = plt.axis()
    plot_pca_projection(data_dir)
    plt.show()

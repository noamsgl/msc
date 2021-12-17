import os
import sys
from tkinter import messagebox
from tkinter.filedialog import askdirectory

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from msc.dataset.dataset import PSPDataset


def plot_pca_projection(data_dir):
    print(f"plotting pca for {data_dir=}")
    # load the data
    dataset = PSPDataset(data_dir)
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
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)

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
    plt.show()


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

plot_pca_projection(data_dir)

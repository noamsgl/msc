import os
import pickle
import re

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm


def save_pattern_plot(X, patient_name, feature_name, window_name, output_path):
    plt.clf()
    plt.title(f"{feature_name}\nfor {patient_name}, {window_name}")
    ax = plt.subplot()
    im = ax.imshow(X)
    plt.xlabel('time (5 s frames)')
    plt.ylabel('index of channel pair')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(output_path)


def imshow_entire_dataset_to_file(dataset_path, patient_name, feature_name):
    images_dir = f"{dataset_path}/images/"
    os.makedirs(images_dir, exist_ok=True)
    samples_df = pd.read_csv(f"{dataset_path}/dataset.csv")
    for root, dirs, files in os.walk(dataset_path):
        for name in tqdm(files, desc="saving images to disk"):
            if os.path.splitext(name)[1] == ".pkl":
                file_path = os.path.join(root, name)
                X = pickle.load(open(file_path, 'rb'))
                window_ids = re.findall('[0-9]+', name)
                assert len(window_ids) == 1, "error: found more than one window id"
                window_id = int(window_ids[0])
                output_path = f"{dataset_path}/images/window_{window_id}.png"
                save_pattern_plot(X, patient_name, feature_name, f"w_{window_id}", output_path)
    return


if __name__ == '__main__':
    dataset_path = r"/cs_storage/noamsi/results/epilepsiae/max_cross_corr/surfCO/pat_3500/20211213T182128/"
    print(f"beginning imshow entire dataset with {dataset_path=}")
    imshow_entire_dataset_to_file(dataset_path, 'pat_3500', feature_name='max_cross_corr')

import os
import pickle
import re
import sys
from tkinter import messagebox
from tkinter.filedialog import askdirectory
from tkinter.simpledialog import askstring

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray
from tqdm import tqdm

from msc.config import get_config


def plot_feature_window(X: ndarray, patient_name: str, feature_name: str, window_name: str, output_path: str=None):
    """
    Saves to disk a plot of a single feature pattern
    Args:
        X:
        patient_name:
        feature_name:
        window_name:
        output_path:

    Returns:

    """
    plt.clf()
    plt.title(f"{feature_name}\nfor {patient_name}, {window_name}")
    ax = plt.subplot()
    im = ax.imshow(X)
    plt.xlabel('time (5 s frames)')
    plt.ylabel('index of channel pair')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def save_pattern_plot(X: ndarray, patient_name: str, feature_name: str, window_name: str, output_path: str):
    """
    Saves to disk a plot of a single feature pattern
    Args:
        X:
        patient_name:
        feature_name:
        window_name:
        output_path:

    Returns:

    """
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


def imshow_entire_dataset_to_file(dataset_path: str, patient_name: str, feature_name: str):
    """
    Converts windows to plots on disk for entire dataset
    Args:
        dataset_path:
        patient_name:
        feature_name:

    Returns:

    """
    images_dir = f"{dataset_path}/images/"
    os.makedirs(images_dir, exist_ok=True)
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
    if not messagebox.askokcancel(title='PSP',
                                  message="Welcome to the Physiological Signal Processing dataset image converter!\n"
                                          "Please select the dataset directory:"):
        sys.exit(-1)

    # get the CurrentStudy dataset directory
    feature_name = askstring('PSP', 'feature_name', initialvalue='phase_lock_val')
    patient_name = askstring('PSP', 'patient_name', initialvalue='pat_3500')
    config = get_config()
    results_dir = config['PATH'][config['RESULTS_MACHINE']]['RESULTS']
    datasets_dir = f"{results_dir}/{config['DATASET']}/{feature_name}/surfCO/{patient_name}"

    init_dir = os.getcwd()
    # show an "Open" dialog box and return the path to the selected file
    data_dir = askdirectory(initialdir=datasets_dir)
    if not data_dir:
        sys.exit(-1)

    print(f"beginning imshow entire dataset with {data_dir=}")
    imshow_entire_dataset_to_file(data_dir, patient_name=patient_name, feature_name=feature_name)

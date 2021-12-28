import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray


def plot_feature_window(X: ndarray, patient_name: str, feature_name: str, window_name: str, output_path: str = None,
                        ax=None):
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
    ax.set_title(f"{feature_name}\nfor {patient_name}, {window_name}")
    im = ax.imshow(X)
    ax.set_xlabel('time (5 s frames)')
    ax.set_ylabel('index of channel pair')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    return None

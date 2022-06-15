import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
from sklearn.utils.estimator_checks import check_estimator

from msc import config
from msc.cache_handler import get_samples_df
from msc.estimators import BSLE

# plt.style.use(['science', 'no-latex'])

SEC = 1
MIN = 60 * SEC
HOUR = 60 * MIN

def plot_calibration_curves(train_X, train_events, test_X, test_times, test_y):
    clf_list = [
        (BSLE(thresh=0.05), "unsupervised"),
        (BSLE(thresh=0.05), "weakly supervised"),
    ]

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(3, 2)
    colors = plt.cm.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}

    # fit estimators to data
    for i, (clf, name) in enumerate(clf_list):
        if name == "unsupervised":
            clf.fit(train_X, prior_events=train_events)
            prob_pred = clf.predict_proba(test_X, samples_times=test_times)
            print("eval_unsupervised_bsle")
            print(brier_score_loss(test_y, prob_pred))
        elif name == "weakly supervised":
            clf.fit(train_X)
            prob_pred = clf.predict_proba(test_X)
            print("eval_weakly_supervised_bsle")
            print(brier_score_loss(test_y, prob_pred))
        else:
            raise ValueError()
        # display calibration curves
        display = CalibrationDisplay.from_predictions(
            test_y,
            prob_pred,
            n_bins=10,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots (Bayesian Seizure Likelihood Estimation)")

    # Add histogram
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for i, (_, name) in enumerate(clf_list):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=10,
            label=name,
            color=colors(i),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.savefig(f"{config['path']['figures']}/bsle/calibration.pdf", bbox_inches='tight')
    # plt.show()

def eval_unsupervised_bsle(train_X, test_X, test_y):
    bsle = BSLE(thresh=0.05)

    bsle.fit(train_X)
    prob_pred = bsle.predict_proba(test_X)
    print("eval_unsupervised_bsle")
    print(brier_score_loss(test_y, prob_pred))

def eval_weakly_supervised_bsle(train_X, train_events, test_X, test_times, test_y):
    bsle = BSLE(thresh=0.05)

    bsle.fit(train_X, y=None, prior_events=train_events)
    prob_pred = bsle.predict_proba(test_X, samples_times=test_times)
    print("eval_weakly_supervised_bsle")
    print(brier_score_loss(test_y, prob_pred))


if __name__ == "__main__":
        
    # bsle = BSLE(thresh=0.4)
    # for estimator, check in check_estimator(bsle, generate_only=True):
    #     check(estimator)

    dataset_id = str(config['dataset_id'])
    t_max = config['t_max']

    # load samples_df
    samples_df = get_samples_df(dataset_id, with_events=True, with_time_to_event=True)

    horizon = 0     # detection
    # horizon = 30 * MIN  # prediction
    samples_df['class'] = samples_df['time_to_event'].apply(lambda x: 1 if x <= horizon else 0)
    samples_df['is_event'] = samples_df['time_to_event'].apply(lambda x: True if x==0 else False)
    
    print(f"class counts is {samples_df['class'].value_counts().to_numpy()}")
    # split train/test
    samples_df['set'] = samples_df['time'].apply(lambda t: 'train' if t < t_max else 'test')

    # get train_times
    train_events = samples_df.loc[(samples_df['is_event']) & (samples_df['set'] == 'train'), 'time']
    # train_events = samples_df.loc[(samples_df['is_event']), 'time']

    # get embeddings
    train_X = np.stack(samples_df.loc[samples_df['set'] == 'train', "embedding"])  # type: ignore
    test_X = np.stack(samples_df.loc[samples_df['set'] == 'test', "embedding"])  # type: ignore
    test_y = samples_df.loc[samples_df['set'] == 'test', "class"].to_numpy()
    test_times = samples_df.loc[samples_df['set'] == 'test', "time"].to_numpy()

    # evaluate unsupervised BSLE
    eval_unsupervised_bsle(train_X, test_X, test_y)
    # evaluate weakly supervised BSLE
    eval_weakly_supervised_bsle(train_X, train_events, test_X, test_times, test_y)
    # plot calibration curves
    plot_calibration_curves(train_X, train_events, test_X, test_times, test_y)

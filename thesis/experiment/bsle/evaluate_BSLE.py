import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, brier_score_loss, roc_auc_score, average_precision_score
from sklearn.utils.estimator_checks import check_estimator

from msc import config
from msc.cache_handler import get_samples_df
from msc.estimators import BSLE
from msc.plot_utils import set_size
from msc.time_utils import SEC, MIN, HOUR
from msc.data_utils import EvalData

plt.style.use(["science", "no-latex"])

fig_width = 478  # pt

def eval_unsupervised_bsle(eval_data):
    bsle = BSLE(thresh=0.05)

    bsle.fit(eval_data.train_X)
    pred_y = bsle.predict(eval_data.test_X)
    plot_auc_roc(eval_data.test_y, pred_y, "unsupervised")
    plot_pr_roc(eval_data.test_y, pred_y, "unsupervised")


def eval_weakly_supervised_bsle(eval_data):
    bsle = BSLE(thresh=0.05)

    bsle.fit(eval_data.train_X, y=None, prior_events=eval_data.train_events)
    pred_y = bsle.predict_proba(eval_data.test_X, samples_times=eval_data.test_times)
    plot_auc_roc(eval_data.test_y, pred_y, "weakly_supervised")
    plot_pr_roc(eval_data.test_y, pred_y, "weakly_supervised")


def plot_auc_roc(test_y, pred_y, fit_mode):
    plt.figure(figsize=set_size(fig_width))
    RocCurveDisplay.from_predictions(test_y, pred_y, ax=plt.gca(), name=fit_mode)
    plt.title("ROC curve for " + fit_mode)
    plt.savefig(
        f"{config['path']['figures']}/bsle/roc_auc_{fit_mode}.pdf", bbox_inches="tight"
    )


def plot_pr_roc(test_y, pred_y, fit_mode):
    plt.figure(figsize=set_size(fig_width))
    PrecisionRecallDisplay.from_predictions(test_y, pred_y, ax=plt.gca(), name=fit_mode)
    plt.title("Precision-Recall curve for " + fit_mode)
    plt.savefig(
        f"{config['path']['figures']}/bsle/pr_auc_{fit_mode}.pdf", bbox_inches="tight"
    )


def plot_calibration_curves(eval_data):
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
            clf.fit(eval_data.train_X, prior_events=eval_data.train_events)
            prob_pred = clf.predict_proba(eval_data.test_X, samples_times=eval_data.test_times)
            print("eval_unsupervised_bsle")
            print(brier_score_loss(eval_data.test_y, prob_pred))
        elif name == "weakly supervised":
            clf.fit(eval_data.train_X)
            prob_pred = clf.predict_proba(eval_data.test_X)
            print("eval_weakly_supervised_bsle")
            print(brier_score_loss(eval_data.test_y, prob_pred))
        else:
            raise ValueError()
        # display calibration curves
        display = CalibrationDisplay.from_predictions(
            eval_data.test_y,
            prob_pred,
            n_bins=10,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title(
        "Calibration plots (Bayesian Seizure Likelihood Estimation)"
    )

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
    plt.savefig(
        f"{config['path']['figures']}/bsle/calibration.pdf", bbox_inches="tight"
    )
    # plt.show()


def plot_brier_score_for_thresholds(eval_data):
    unsupervised_scores = []
    weakly_supervised_scores = []
    thresholds = np.linspace(0, 0.2, 21, endpoint=True)
    for threshold in tqdm(thresholds):
        bsle = BSLE(thresh=threshold)
        # unsupervised fit
        bsle.fit(eval_data.train_X)
        pred_y = bsle.predict(eval_data.test_X)
        score = brier_score_loss(eval_data.test_y, pred_y)
        unsupervised_scores.append(score)
        # weakly supervised fit
        bsle.fit(eval_data.train_X, y=None, prior_events=eval_data.train_events)
        pred_y = bsle.predict_proba(eval_data.test_X, samples_times=eval_data.test_times)
        score = brier_score_loss(eval_data.test_y, pred_y)
        weakly_supervised_scores.append(score)
    plt.figure(figsize=set_size(fig_width))
    plt.title("Brier Score for different thresholds")
    plt.xlabel("threshold")
    plt.ylabel("Brier Score")
    plt.plot(thresholds, unsupervised_scores, label="unsupervised")
    plt.plot(thresholds, weakly_supervised_scores, label="weakly supervised")
    plt.legend()
    plt.savefig(
        f"{config['path']['figures']}/bsle/brier_score_for_thresholds.pdf",
        bbox_inches="tight",
    )


def plot_roc_auc_score_for_thresholds(eval_data):
    unsupervised_scores = []
    weakly_supervised_scores = []
    thresholds = np.linspace(0, 0.2, 21, endpoint=True)
    for threshold in tqdm(thresholds):
        bsle = BSLE(thresh=threshold)
        # unsupervised fit
        bsle.fit(eval_data.train_X)
        pred_y = bsle.predict(eval_data.test_X)
        score = roc_auc_score(eval_data.test_y, pred_y)
        unsupervised_scores.append(score)
        # weakly supervised fit
        bsle.fit(eval_data.train_X, y=None, prior_events=eval_data.train_events)
        pred_y = bsle.predict_proba(eval_data.test_X, samples_times=eval_data.test_times)
        score = roc_auc_score(eval_data.test_y, pred_y)
        weakly_supervised_scores.append(score)
    plt.figure(figsize=set_size(fig_width))
    plt.title("ROC-AUC for different thresholds")
    plt.xlabel("threshold")
    plt.ylabel("ROC-AUC")
    plt.plot(thresholds, unsupervised_scores, label="unsupervised")
    plt.plot(thresholds, weakly_supervised_scores, label="weakly supervised")
    plt.legend()
    plt.savefig(
        f"{config['path']['figures']}/bsle/roc_auc_score_for_thresholds.pdf",
        bbox_inches="tight",
    )
    

def plot_average_precision_score_for_thresholds(eval_data):
    unsupervised_scores = []
    weakly_supervised_scores = []
    thresholds = np.linspace(0, 0.2, 21, endpoint=True)
    for threshold in tqdm(thresholds):
        bsle = BSLE(thresh=threshold)
        # unsupervised fit
        bsle.fit(eval_data.train_X)
        pred_y = bsle.predict(eval_data.test_X)
        score = average_precision_score(eval_data.test_y, pred_y)
        unsupervised_scores.append(score)
        # weakly supervised fit
        bsle.fit(eval_data.train_X, y=None, prior_events=eval_data.train_events)
        pred_y = bsle.predict_proba(eval_data.test_X, samples_times=eval_data.test_times)
        score = average_precision_score(eval_data.test_y, pred_y)
        weakly_supervised_scores.append(score)
    plt.figure(figsize=set_size(fig_width))
    plt.title("Average Precision for different thresholds")
    plt.xlabel("threshold")
    plt.ylabel("AP")
    plt.plot(thresholds, unsupervised_scores, label="unsupervised")
    plt.plot(thresholds, weakly_supervised_scores, label="weakly supervised")
    plt.legend()
    plt.savefig(
        f"{config['path']['figures']}/bsle/average_precision_score_for_thresholds.pdf",
        bbox_inches="tight",
    )


def single_horizon_plots(samples_df, horizon=0):
    samples_df["class"] = samples_df["time_to_event"].apply(
        lambda x: 1 if x <= horizon else 0
    )

    # get training event times
    train_events = samples_df.loc[
        (samples_df["is_event"]) & (samples_df["set"] == "train"), "time"
    ].to_numpy()

    # get train/test data
    train_X = np.stack(samples_df.loc[samples_df["set"] == "train", "embedding"])  # type: ignore
    test_X = np.stack(samples_df.loc[samples_df["set"] == "test", "embedding"])  # type: ignore
    test_y = samples_df.loc[samples_df["set"] == "test", "class"].to_numpy()
    test_times = samples_df.loc[samples_df["set"] == "test", "time"].to_numpy()
    
    eval_data = EvalData(train_X, train_events, test_X, test_times, test_y)

    plot_brier_score_for_thresholds(eval_data)
    plot_average_precision_score_for_thresholds(eval_data)
    plot_roc_auc_score_for_thresholds(eval_data)
    eval_unsupervised_bsle(eval_data)
    eval_weakly_supervised_bsle(eval_data)
    # plot_calibration_curves(eval_data)

def get_roc_auc_scores_for_thresholds(eval_data, thresholds):
    unsupervised_scores = []
    weakly_supervised_scores = []
    for threshold in thresholds:
        bsle = BSLE(thresh=threshold)
        # unsupervised fit
        bsle.fit(eval_data.train_X)
        pred_y = bsle.predict(eval_data.test_X)
        score = roc_auc_score(eval_data.test_y, pred_y)
        unsupervised_scores.append(score)
        # weakly supervised fit
        bsle.fit(eval_data.train_X, y=None, prior_events=eval_data.train_events)
        pred_y = bsle.predict_proba(eval_data.test_X, samples_times=eval_data.test_times)
        score = roc_auc_score(eval_data.test_y, pred_y)
        weakly_supervised_scores.append(score)
    return unsupervised_scores, weakly_supervised_scores


def auc_roc_scores_for_thresholds_and_horizons(samples_df, time_scale):
    if time_scale == "sec":
        horizons = np.arange(0, 65, 5) * SEC
    elif time_scale == "min":
        horizons = np.arange(0, 65, 5) * MIN
    else:
        raise ValueError("time_scale must be either SEC or MIN")
    thresholds = np.linspace(0, 0.2, 21, endpoint=True)
    unsupervised_score_rows = []
    weakly_supervised_score_rows = []
    for horizon in horizons:
        samples_df["class"] = samples_df["time_to_event"].apply(
        lambda x: 1 if x <= horizon else 0
        )
        
        # get training event times
        train_events = samples_df.loc[
            (samples_df["is_event"]) & (samples_df["set"] == "train"), "time"
        ].to_numpy()

        # train/test data
        train_X = np.stack(samples_df.loc[samples_df["set"] == "train", "embedding"])  # type: ignore
        test_X = np.stack(samples_df.loc[samples_df["set"] == "test", "embedding"])  # type: ignore
        test_y = samples_df.loc[samples_df["set"] == "test", "class"].to_numpy()
        test_times = samples_df.loc[samples_df["set"] == "test", "time"].to_numpy()

        eval_data = EvalData(train_X, train_events, test_X, test_times, test_y)

        unsupervised_scores, weakly_supervised_scores = get_roc_auc_scores_for_thresholds(eval_data, thresholds)
        unsupervised_score_rows.append(unsupervised_scores)
        weakly_supervised_score_rows.append(weakly_supervised_scores)
    unsupervised_scores = np.stack(unsupervised_score_rows).T
    weakly_supervised_scores = np.stack(weakly_supervised_score_rows).T
    row_labels = [f"{t:.2f}" for t in thresholds]
    col_labels = [f"{int(h/MIN)}" for h in horizons] if time_scale == "min" else [f"{int(h/SEC)}" for h in horizons]

    # set up subplot axes
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=set_size(fig_width))
    im = axes[0].imshow(unsupervised_scores, cmap="RdBu", vmin=0, vmax=1)
    axes[0].set_title("Unsupervised")
    axes[0].set_xticks(np.arange(unsupervised_scores.shape[1]), labels=col_labels)
    axes[0].set_yticks(np.arange(unsupervised_scores.shape[0]), labels=row_labels)
    axes[0].set_xlabel(f"Horizon ({time_scale})")
    axes[0].set_ylabel("Threshold")

    im = axes[1].imshow(weakly_supervised_scores, cmap="RdBu", vmin=0, vmax=1)
    axes[1].set_title("Weakly Supervised")
    axes[1].set_xticks(np.arange(unsupervised_scores.shape[1]), labels=col_labels)
    axes[1].set_yticks(np.arange(unsupervised_scores.shape[0]), labels=row_labels)
    axes[1].set_xlabel(f"Horizon ({time_scale})")
    axes[1].set_ylabel("Threshold")
    # create colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("ROC AUC", rotation=-90, va="bottom")

    # rotate the xtick labels
    plt.setp(axes[0].get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    plt.setp(axes[1].get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    plt.savefig(
        f"{config['path']['figures']}/bsle/auc_roc_scores_for_thresholds_and_horizons_{time_scale}.pdf",
        bbox_inches="tight",
    )
    
if __name__ == "__main__":
    # bsle = BSLE()
    # for estimator, check in check_estimator(bsle, generate_only=True):
    #     check(estimator)

    dataset_id = str(config["dataset_id"])
    t_max = config["t_max"]

    # load samples_df
    samples_df = get_samples_df(dataset_id, with_events=True, with_time_to_event=True)
    
    # split train/test
    samples_df["set"] = samples_df["time"].apply(
        lambda t: "train" if t < t_max else "test"
    )
    
    # add is_event
    samples_df["is_event"] = samples_df["time_to_event"].apply(
        lambda x: True if x == 0 else False
    )

    # define horizon
    horizon = 0  # detection
    # horizon = 5 * SEC
    # horizon = 30 * MIN  # prediction

    single_horizon_plots(samples_df, horizon)
    auc_roc_scores_for_thresholds_and_horizons(samples_df, "sec")
    auc_roc_scores_for_thresholds_and_horizons(samples_df, "min")

import matplotlib.pyplot as plt
import pandas as pd

from msc.config import get_config


def plot_results_score(results_row, patient_name, feature_name, score=('precision'), logy=False, ax=None, color=None):
    """Plot a Bar Plot of classification results
    """
    if ax is None:
        fig, ax = plt.figure()
    scoring_cols = [f'test_{score}']
    results = results_row.loc[:, ['classifier_name'] + scoring_cols]
    means = results.groupby('classifier_name').mean()
    errors = results.groupby('classifier_name').std()
    return means.plot.bar(ax=ax, yerr=errors, xlabel='Classifier Name', logy=logy, rot=0, grid=True, color=color,
                          title=f"{score} for Patient {patient_name}, Feature {feature_name}")


def plot_results_time(results_row, patient_name, feature_name, time_col=('score_time'), logy=False, ax=None,
                      color=None):
    if ax is None:
        fig, ax = plt.figure()
    results = results_row.loc[:, ['classifier_name'] + [time_col]]
    means = results.groupby('classifier_name').mean()
    errors = results.groupby('classifier_name').std()
    return means.plot.bar(ax=ax, yerr=errors, xlabel='Classifier Name', ylabel='time (seconds)', logy=logy, rot=0,
                          grid=True, color=color,
                          title=f"{time_col} for Patient {patient_name}, Feature {feature_name}")


def plot_results_scores(results_row, patient_name, feature_name, ax, scorings=('precision', 'recall', 'roc_auc')):
    scoring_cols = [f'test_{sc}' for sc in scorings]
    results = results_row.loc[:, ['classifier_name'] + scoring_cols]
    means = results.groupby('classifier_name').mean()
    errors = results.groupby('classifier_name').std()
    return means.plot.bar(ax=ax, yerr=errors, xlabel='Classifier Name', ylim=(0, 1),
                          title=f"Evaluation Results for Different Classifiers\n{patient_name}, {feature_name}", rot=18)


def plot_results_times(results_row, patient_name, feature_name, ax, timing_cols=('fit_time', 'score_time')):
    results = results_row.loc[:, ['classifier_name'] + list(timing_cols)]
    means = results.groupby('classifier_name').mean()
    errors = results.groupby('classifier_name').std()
    return means.plot.bar(ax=ax, yerr=errors, xlabel='Classifier Name',
                          title=f"Computation Time for Different Classifiers\n{patient_name}, {feature_name}", rot=18)


if __name__ == '__main__':
    config = get_config()
    results_fpath = f"{config['PATH']['LOCAL']['RESULTS']}/complete_results.csv"
    results = pd.read_csv(results_fpath)

    selected_patient = "pat_7200"


    # selected_feature = "spect_corr"
    #
    # results = results.set_index(['patient_name', 'feature_name']).loc[selected_patient, selected_feature]
    # fig, ax = plt.subplots(1)
    # plot_results_times(results, selected_patient, selected_feature, ax=ax)
    # # plot_results_scores(results, selected_patient, selected_feature, scorings=('roc_auc',), ax=ax)
    #
    # plt.tight_layout()
    #
    # plt.show()

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    selected_features = ['max_cross_corr', 'phase_lock_val', 'spect_corr', 'time_corr']
    results = results.loc[results.feature_name.apply(lambda val: val in selected_features)]
    # create plot
    # ax = plt.axes()
    # ax.set_title(f'ROC AUC for patient {pat_number}')
    # sns.heatmap(frame, annot=True, ax=ax)
    # ax.figure.tight_layout()
    # plt.show()
    pat_numbers = [3500, 3700, 7200]

    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)

    fig, axes = plt.subplots(1, 3, figsize=(13.33, 7.5), dpi=180, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .28, .03, .6])
    fig.suptitle("Training Time", fontsize=21)
    # fig.text(0.5, 0.04, 'Feature', ha='center')
    # fig.text(0.04, 0.5, 'Classifier', va='center', rotation='vertical')
    for i, ax in enumerate(axes.flat):
        pat_number = pat_numbers[i]

        selected_patient = f"pat_{pat_number}"

        # build results frame
        frame = results.set_index('patient_name').loc[selected_patient].groupby(
            ['classifier_name', 'feature_name']).aggregate('mean').test_roc_auc.unstack(0)
        ax.set_title(f'patient {pat_number}', fontsize=16)
        sns.heatmap(frame.transpose(),
                    cmap=sns.color_palette('Spectral', as_cmap=True),
                    annot=True, ax=ax, cbar=i == 0,
                    vmin=0, vmax=1, cbar_ax=None if i else cbar_ax, annot_kws={"fontsize": 12}, )
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.figure.tight_layout()
    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.show()
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

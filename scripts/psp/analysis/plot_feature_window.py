import matplotlib.pyplot as plt
import pandas as pd


def plot_results_score(results_row, patient_name, feature_name, score=('precision'), logy=False, ax=None, color=None):
    if ax is None:
        fig, ax = plt.figure()
    scoring_cols = [f'test_{score}']
    results = results_row.loc[:, ['classifier_name'] + scoring_cols]
    means = results.groupby('classifier_name').mean()
    errors = results.groupby('classifier_name').std()
    return means.plot.bar(ax=ax, yerr=errors, xlabel='Classifier Name', logy=logy, rot=0, grid=True, color=color,
                          title=f"{score} for Patient {patient_name}, Feature {feature_name}")


def plot_results_time(results_row, patient_name, feature_name, time_col=('score_time'), logy=False, ax=None, color=None):
    if ax is None:
        fig, ax = plt.figure()
    results = results_row.loc[:, ['classifier_name'] + [time_col]]
    means = results.groupby('classifier_name').mean()
    errors = results.groupby('classifier_name').std()
    return means.plot.bar(ax=ax, yerr=errors, xlabel='Classifier Name', ylabel='time (seconds)', logy=logy, rot=0, grid=True, color=color,
                          title=f"{time_col} for Patient {patient_name}, Feature {feature_name}")


def plot_results_scores(results_row, ax, scorings=('precision', 'recall', 'roc_auc'), ):
    scoring_cols = [f'test_{sc}' for sc in scorings]
    results = results_row.loc[:, ['classifier_name'] + scoring_cols]
    means = results.groupby('classifier_name').mean()
    errors = results.groupby('classifier_name').std()
    return means.plot.bar(ax=ax, yerr=errors, xlabel='Classifier Name',
                          title='Evaluation Results for Different Classifiers', rot=12)


def plot_results_times(results_fpath, timing_cols=('fit_time', 'score_time')):
    results = pd.read_csv(results_fpath)
    results = results.loc[:, ['name'] + list(timing_cols)]
    means = results.groupby('name').mean()
    errors = results.groupby('name').std()
    return means.plot.bar(yerr=errors, xlabel='Classifier Name', title='Computation Time for Different Classifiers')


if __name__ == '__main__':
    results_fpath = r"C:\Users\noam\Repositories\noamsgl\msc\scripts\psp\training\results_2.csv"
    results = pd.read_csv(results_fpath)

    plot_results_times(results)
    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

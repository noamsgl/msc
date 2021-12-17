import matplotlib.pyplot as plt
import pandas as pd


def plot_results_scores(results_path, scorings=('precision', 'recall', 'roc_auc')):
    results = pd.read_csv(results_path)
    scoring_cols = [f'test_{sc}' for sc in scorings]
    results = results.loc[:, ['name'] + scoring_cols]

    means = results.groupby('name').mean()
    errors = results.groupby('name').std()
    return means.plot.bar(yerr=errors, xlabel='Classifier Name', title='Evaluation Results for Different Classifiers')

def plot_results_times(results_path, timing_cols=('fit_time', 'score_time')):
    results = pd.read_csv(results_path)
    results = results.loc[:, ['name'] + list(timing_cols)]

    means = results.groupby('name').mean()
    errors = results.groupby('name').std()
    return means.plot.bar(yerr=errors, xlabel='Classifier Name', title='Computation Time for Different Classifiers')



path = r"C:\Users\noam\Repositories\noamsgl\msc\scripts\psp\training\results_2.csv"

plot_results_times(path)
plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

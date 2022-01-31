import matplotlib.pyplot as plt
import pandas as pd

from msc.config import get_config


def plot_results_scores(results_row, patient_name, classifier_name, ax, scorings=('precision', 'recall', 'roc_auc')):
    scoring_cols = [f'test_{sc}' for sc in scorings]
    results = results_row.loc[:, ['feature_name'] + scoring_cols]
    means = results.groupby('feature_name').mean()
    errors = results.groupby('feature_name').std()
    return means.plot.bar(ax=ax, yerr=errors, xlabel='Feature Name', ylim=(0, 1),
                          title=f"Evaluation Results for Different Features\n{patient_name}, {classifier_name}", rot=18)


if __name__ == '__main__':
    config = get_config()
    results_fpath = f"{config['PATH']['LOCAL']['RESULTS']}/complete_results.csv"
    results = pd.read_csv(results_fpath)

    selected_patient = "pat_7200"
    selected_classifier = "Linear SVM"

    results = results.set_index(['patient_name', 'classifier_name']).loc[selected_patient, selected_classifier]
    fig, ax = plt.subplots(1)
    plot_results_scores(results, selected_patient, selected_classifier, scorings=('roc_auc',), ax=ax)

    plt.tight_layout()

    plt.show()

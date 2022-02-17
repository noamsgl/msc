"""
Noam Siegel
gp_analyze_inference.py

Get results for requested_params.
Plot jointplot of two params.
Report results_df and plots ClearML
"""

import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task

from msc.results_collectors import GPResultsCollector

if __name__ == '__main__':
    task = Task.init(project_name=f"analyze", task_name=f"gp_matern_params")
    requested_params = ['covar_module.raw_outputscale', 'covar_module.base_kernel.raw_lengthscale']

    results_df = GPResultsCollector(requested_params).results_df

    logger = task.get_logger()
    logger.report_table("results_df", "Results Table", table_plot=results_df)
    logger.report_scalar('num_samples', 'num_samples', len(results_df), iteration=0)
    sns.jointplot(data=results_df,
                  x=requested_params[0], y=requested_params[1],
                  hue="label_desc", palette="muted")
    plt.suptitle("Matern Kernel Params for Dog_1 Dataset")
    plt.show()

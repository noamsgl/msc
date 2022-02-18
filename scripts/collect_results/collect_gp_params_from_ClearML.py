import datetime

from msc.results_collectors import GPResultsCollector

if __name__ == '__main__':
    # get results of multitask (pair) GP params MLE
    requested_project_name = "inference/pairs/Dog_1"
    requested_params = ['covar_module.data_covar_module.base_kernel.raw_lengthscale',
                        'covar_module.data_covar_module.raw_lengthscale',
                        'covar_module.task_covar_module.covar_factor[0]',
                        'covar_module.task_covar_module.covar_factor[1]',
                        'covar_module.task_covar_module.raw_var[0]',
                        'covar_module.task_covar_module.raw_var[1]']

    split_date = datetime.datetime(year=2022, month=2, day=10)
    results_df = GPResultsCollector.from_clearml(requested_project_name, requested_params,
                                    split_version_by_date=split_date).results_df

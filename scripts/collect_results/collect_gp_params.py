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

    # requested_params_new = ['covar_module.data_covar_module.raw_lengthscale',
    #                         'covar_module.task_covar_module.covar_factor[0]',
    #                         'covar_module.task_covar_module.covar_factor[1]',
    #                         'covar_module.task_covar_module.raw_var[0]',
    #                         'covar_module.task_covar_module.raw_var[1]']

    results_df = GPResultsCollector(requested_project_name, requested_params).results_df
    results_df['version'] = results_df['completed'].apply(
        lambda dt: 0 if dt.replace(tzinfo=None) < datetime.datetime(year=2022, month=2, day=10) else 1)

    # merge version 0 params into version 1 params
    results_df.loc[results_df[
                       'covar_module.data_covar_module.raw_lengthscale'].isna(), 'covar_module.data_covar_module.raw_lengthscale'] = \
        results_df['covar_module.data_covar_module.base_kernel.raw_lengthscale']
    results_df = results_df.drop(columns=['covar_module.data_covar_module.base_kernel.raw_lengthscale'])

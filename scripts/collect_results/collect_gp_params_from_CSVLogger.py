from msc.results_collectors import GPResultsCollector

if __name__ == '__main__':
    # define requested parameters
    requested_params = ['covar_module.data_covar_module.raw_lengthscale',
                        'covar_module.task_covar_module.covar_factor[0]',
                        'covar_module.task_covar_module.covar_factor[1]',
                        'covar_module.task_covar_module.raw_var[0]',
                        'covar_module.task_covar_module.raw_var[1]']

    logs_dir = r"C:\Users\noam\Repositories\noamsgl\msc\results\lightning_logs_from_cluster"

    results_df = GPResultsCollector.from_csv_logs(logs_dir, requested_params).results_df

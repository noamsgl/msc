"""
gp_analyze_inference.py

Gaussian Process Analyze Inference

1) load the
2) create sns.jointplot()



"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from clearml import Task
from clearml.backend_api import Session
from clearml.backend_api.services import tasks, projects
from requests import Request


class GPResultsCollector:
    def __init__(self, requested_params):
        # create an authenticated session
        session = Session()

        # get projects matching the name 'inference/'
        req = Request('POST')
        projects_res = session.send(projects.GetAllRequest(name='inference'))
        # get all the project Ids matching the project name 'inference'
        project_ids = [p.id for p in projects_res.response.projects]

        projects_df = pd.DataFrame([p.to_dict() for p in projects_res.response.projects])
        projects_df["label_desc"] = projects_df["name"].apply(
            lambda name: 'interictal' if 'interictal' in name else 'ictal')

        # get all the tasks
        tasks_list = []
        for i in range(5):
            print(f"getting page={i}")
            tasks_res = session.send(tasks.GetAllRequest(project=project_ids, page_size=500, page=i))
            tasks_list += tasks_res.response_data['tasks']

        results_df = pd.DataFrame(tasks_list)
        results_df = self.parse_last_metrics(results_df, requested_params)

        results_df = pd.merge(results_df, projects_df[['id', 'name', 'label_desc']], left_on='project', right_on='id',
                              suffixes=('_task', '_project'))
        self.results_df = results_df.dropna(subset=requested_params)

    @staticmethod
    def parse_last_metrics(results_df, requested_params):
        """last_metrics is a field returned by the ClearML API. This function parses the field results and appends requested_params values to results_df"""

        def get_param_from_last_metrics(last_metrics: dict, param_name):
            for value in last_metrics.values():
                assert isinstance(value, dict), "Error: last_metrics value entries should be of type dict"
                if len(value) > 1:
                    # we know the reported parameters only have one variant
                    continue
                values_dict = list(value.values())[0]
                if param_name in values_dict['metric']:
                    return values_dict['value']
            return None

        for param_name in requested_params:
            results_df[param_name] = results_df["last_metrics"].apply(get_param_from_last_metrics,
                                                                      param_name=param_name)

        return results_df


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

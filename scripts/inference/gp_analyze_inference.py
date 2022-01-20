"""
gp_analyze_inference.py

Gaussian Process Analyze Inference

1) load the
2) create sns.jointplot()



"""
import pandas as pd
from clearml import Task
from clearml.backend_api import Session
from requests import Request
from clearml.backend_api.services import tasks, events, projects
import seaborn as sns
import matplotlib.pyplot as plt

from msc.dataset import DogDataset
from msc.models import SingleSampleEEGGPModel




class GPResultsCollector:
    def __init__(self, requested_params):
        # create an authenticated session
        session = Session()

        # get projects matching the name 'inference/'
        req = Request('POST')
        res = session.send(projects.GetAllRequest(name='inference'))
        # get all the project Ids matching the project name 'inference'
        project_ids = [p.id for p in res.response.projects]

        # get all the tasks
        tasks_list = []
        for i in range(5):
            print(f"getting page={i}")
            res = session.send(tasks.GetAllRequest(project=project_ids, page_size=500, page=i))
            tasks_list += res.response_data['tasks']

        results_df = pd.DataFrame(tasks_list)
        results_df = self.parse_last_metrics(results_df, requested_params)

        projs_dfs = []

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
            results_df[param_name] = results_df["last_metrics"].apply(get_param_from_last_metrics, param_name=param_name)

        return results_df


if __name__ == '__main__':
    requested_params = ['covar_module.raw_outputscale', 'covar_module.base_kernel.raw_lengthscale']
    results_df = GPResultsCollector(requested_params)
    sns.jointplot(data=results_df,
                  x=requested_params[0], y=requested_params[1],
                  hue="label_desc", palette="muted")
    plt.show()

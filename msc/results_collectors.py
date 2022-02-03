"""
Noam Siegel
A set of classes to collect results from ClearML server
"""

import pandas as pd

from clearml.backend_api import Session
from clearml.backend_api.services import tasks, projects
from requests import Request


class GPResultsCollector:
    """
    1) get Gaussian Process related requested_params from ClearML
    2) parse results into results_df
    """

    def __init__(self, requested_project_name="inference", requested_params=None, n_pages=5):
        """

        Args:
            requested_project_name:
            requested_params:
            n_pages: number of ClearML results pages to request. Each page contains 500 results.
        """
        assert requested_params is not None, "error: requested params must be nonempty iterable of param names"
        # create an authenticated session
        session = Session()

        # get projects matching the name `requested_project_name`
        projects_res = session.send(projects.GetAllRequest(name=requested_project_name))
        # get all the project Ids matching the project name 'inference'
        project_ids = [p.id for p in projects_res.response.projects]
        # concat all results to dataframe
        projects_df = pd.DataFrame([p.to_dict() for p in projects_res.response.projects])
        # add label_desc based on file name
        projects_df["label_desc"] = projects_df["name"].apply(
            lambda name: 'interictal' if 'interictal' in name else 'ictal')

        # get all the tasks
        tasks_list = []
        for i in range(n_pages):
            print(f"getting page={i + 1}/{n_pages}")
            tasks_res = session.send(tasks.GetAllRequest(project=project_ids, page_size=500, page=i))
            tasks_list += tasks_res.response_data['tasks']

        results_df = pd.DataFrame(tasks_list)
        results_df = self.parse_last_metrics(results_df, requested_params)

        results_df = pd.merge(results_df, projects_df[['id', 'name', 'label_desc']], left_on='project', right_on='id',
                              suffixes=('_task', '_project'))

        # keep only status==completed
        results_df = results_df.loc[results_df['status'] == 'completed']

        # drop NaNs
        results_df = results_df.dropna(subset=requested_params)
        # save to self
        self.results_df = results_df

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

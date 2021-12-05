import configparser
import os.path

import git


def get_config():
    # read local file `config.ini` from root of git repository
    repo = git.Repo('.', search_parent_directories=True)
    root_dir = repo.working_tree_dir
    config = configparser.ConfigParser()
    config.read(f'{root_dir}/settings/config.ini')
    return config
import configparser
import os.path

import git
from git import InvalidGitRepositoryError


def get_config():
    # read local file `config.ini` from root_dir. If root_dir not given, will use root of git repository
    try:
        repo = git.Repo('.', search_parent_directories=True)
        root_dir = repo.working_tree_dir
    except InvalidGitRepositoryError as e:
        root_dir = "../.."
    config = configparser.ConfigParser()
    config.read(f'{root_dir}/settings/config.ini')
    return config
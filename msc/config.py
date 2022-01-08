from configparser import ConfigParser

import git
import yaml
from git import InvalidGitRepositoryError


def get_config(kind='yaml') -> ConfigParser:
    # read local file `config.yaml` from root_dir. If root_dir not given, will use root of git repository
    try:
        repo = git.Repo('config', search_parent_directories=True)
        root_dir = repo.working_tree_dir
    except InvalidGitRepositoryError:
        root_dir = ".."

    if kind == 'yaml':
        config = yaml.safe_load(open(f'{root_dir}/settings/config.yaml', 'r'))
        return config

    else:
        raise ValueError("incorrect config kind")

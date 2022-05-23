import os
from configparser import ConfigParser
from typing import Tuple

import git
import yaml
from git import InvalidGitRepositoryError


def get_config(kind='yaml') -> ConfigParser:
    # read local file `config.yaml` from config dir.
    try:
        repo = git.Repo(search_parent_directories=True)
        root_dir = repo.working_tree_dir
    except InvalidGitRepositoryError:
        root_dir = ".."

    if kind == 'yaml':
        config = yaml.safe_load(open(f'{root_dir}/config/config.yaml', 'r'))
        return config

    else:
        raise ValueError("incorrect config kind")


def get_authentication() -> Tuple[str, str]:
    # read local file `authentication.yaml` from config dir.
    try:
        repo = git.Repo(search_parent_directories=True)
        root_dir = repo.working_tree_dir
    except InvalidGitRepositoryError:
        root_dir = ".."

    authentication_fpath = f'{root_dir}/config/authentication.yaml'
    assert os.path.isfile(authentication_fpath), "error: authentication.yaml file not found in config directory"
    with open(authentication_fpath, 'r') as f:
        config = yaml.safe_load(f)
    return config['USER'], config['PASSWORD']

import configparser
from configparser import ConfigParser

import git
import yaml
from git import InvalidGitRepositoryError


def get_config(kind='yaml') -> ConfigParser:
    # read local file `config.ini` from root_dir. If root_dir not given, will use root of git repository
    try:
        repo = git.Repo('.', search_parent_directories=True)
        root_dir = repo.working_tree_dir
    except InvalidGitRepositoryError as e:
        root_dir = "../.."

    if kind == 'yaml':
        config = yaml.safe_load(open(f'{root_dir}/settings/config.yaml', 'r'))
        return config

    elif kind == 'ini':
        config = configparser.ConfigParser()
        return config.read(f'{root_dir}/settings/config.ini')

    else:
        raise ValueError("incorrect config kind")

import os

import paramiko

from msc.config import get_config
from .scp import SCPClient


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def data2headpath(datapath):
    filepath, ext = os.path.splitext(datapath)
    headpath = filepath + '.head'
    return headpath

def download_file_scp(relative_file_path):
    """
    Downloads the .data and .head of a file from a remote location using scp.
    Configure the remote location and local save location in the config file.
    Gets SSH settings from environment variables MSC_SERVER, MSC_USER, MSC_TOKEN
    Args:
        relative_file_path: a .data file

    Returns: local_file_path
    """
    # load paths from config file


    config = get_config()
    datasets_path_remote = config.get('DATA', 'DATASETS_PATH_REMOTE')
    datasets_path_local = config.get('DATA', 'DATASETS_PATH_LOCAL')
    dataset = config.get('DATA', 'DATASET')

    # make remote and local paths
    relative_file_dir, filename = os.path.split(relative_file_path)
    local_save_dir = f"{datasets_path_local}/{dataset}/{relative_file_dir}"
    remote_fetch_path = f"{datasets_path_remote}/{dataset}/{relative_file_path}"

    local_file_path = f"{local_save_dir}/{filename}"
    file_exists = os.path.isfile(local_file_path) & os.path.isfile(data2headpath(local_file_path))
    if file_exists:
        print(f"file exists locally at {local_file_path}")
        return local_file_path

    else:
        # get ssh settings from environment variables
        server = os.getenv('MSC_SERVER')
        port = 22
        user = os.getenv('MSC_USER')
        password = os.getenv('MSC_TOKEN')

        # instantiate SSH and SCP session
        ssh = createSSHClient(server, port, user, password)
        scp = SCPClient(ssh.get_transport())

        # make local dir
        os.makedirs(local_save_dir, exist_ok=True)
        print(f"attempting to download {remote_fetch_path} to {local_save_dir}")
        scp.get(remote_path=remote_fetch_path, local_path=local_save_dir)
        print(f"success")
        print(f"attempting to download {data2headpath(remote_fetch_path)} to {local_save_dir}")
        scp.get(remote_path=data2headpath(remote_fetch_path), local_path=local_save_dir)
        print(f"success")
        return local_file_path

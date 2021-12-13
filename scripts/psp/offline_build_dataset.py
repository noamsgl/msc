from datetime import datetime
import os

import sys

from msc.config import get_config

config = get_config()
results_dir = f"{config.get('RESULTS', 'RESULTS_DIR')}"
iso_8601_format = '%Y%m%dT%H%M%S'  # e.g., 20211119T221000
results_path = f"{results_dir}/results_{datetime.now().strftime(iso_8601_format)}.txt"
os.system("nohup bash -c '" +
          sys.executable + " build_dataset.py > result.txt" +
          "' &")
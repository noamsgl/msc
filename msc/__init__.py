__version__ = "0.1.0a1"
#
from .config import get_config
from .experiments import get_exps

config = get_config()
exps = get_exps()

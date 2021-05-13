from pkg_resources import get_distribution

__version__ = get_distribution('ds_tools').version

from .exploration import *
from .evaluation import *
from .miscs import *
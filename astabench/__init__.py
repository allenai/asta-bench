from importlib.metadata import version as get_version

from . import evals, solvers
from .providers.providers import openai_json

__version__ = get_version("astabench")

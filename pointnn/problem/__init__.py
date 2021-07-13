from . import base
from . import starcraft
from . import weather
from . import traffic


def make_problem(problem_name, kwargs):
    # TODO: Error reporting?
    return base._problem_map[problem_name](**kwargs)

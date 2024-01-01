"""This file contains all the probabilistic functions used by the solver."""
from typing import Union

import numpy as np


BOLTZMANN_CONSTANT: float = 1.380649


def exponential(delta_energy, temp) -> Union[int, float]:
    if delta_energy < 0:
        return 1
    return np.exp(-delta_energy / (BOLTZMANN_CONSTANT * temp))

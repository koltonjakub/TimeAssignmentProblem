import numpy as np
from enum import Enum
from typing import Callable
import matplotlib.pyplot as plt


class CoolingTypes(Enum):
    LIN = 'linear',
    POLY = 'polynomial',
    EXP = 'exponential',
    LOG = 'logarithmic'


def linear_cooling_init(init_temp: float, final_temp: float, max_iter: int, iters: int = 1) -> Callable[
    [float, int], float]:
    def linear_cooling(temp: float, it: int):
        if it == 0:
            return init_temp
        if it % iters != 0:
            return temp
        series = (max_iter + iters) // iters - 1
        a = (init_temp - final_temp) / series
        return final_temp + a * (series - it // iters)
    return linear_cooling


def polynomial_cooling_init(init_temp: float, final_temp: float, max_iter: int, iters: int = 1, poly=None) -> Callable[
    [float, int], float]:
    def polynomial_cooling(temp: float, it: int):
        if it == 0:
            return init_temp
        if it % iters != 0:
            return temp
        series = (max_iter + iters) // iters - 1
        a = (init_temp - final_temp) / (series ** poly)
        return final_temp + a * (series - it // iters) ** (poly)
    return polynomial_cooling


def exponential_cooling_init(init_temp: float, final_temp: float, max_iter: int, iters: int = 1, poly=None) -> Callable[
    [float, int], float]:
    def exponential_cooling(temp: float, it: int):
        if it == 0:
            return init_temp
        if it % iters != 0:
            return temp
        return final_temp + (init_temp - final_temp) * np.exp(-it // iters)
    return exponential_cooling


def logarithmic_cooling_init(init_temp: float, final_temp: float, max_iter: int, iters: int = 1, poly=None) -> Callable[
    [float, int], float]:
    def logarithmic_cooling(temp: float, it: int):
        if it == 0:
            return init_temp
        if it % iters != 0 or temp == 0:
            return temp
        series = (max_iter + iters) // iters - 1
        a = (init_temp - final_temp) / np.log(series)
        return final_temp + a * np.log(series - it // iters)
    return logarithmic_cooling


def cooling_factory(init_temp: float, final_temp: float, max_iter: int, cool_type: CoolingTypes, iters: int = 1,
                    poly: int = None) -> Callable[[float, int], float]:
    """
        Function which creates cooling function.
        @param init_temp: initial temperature
        @type init_temp: float
        @param final_temp: final temperature
        @type  final_temp: float
        @param max_iter: number of iterations
        @type max_iter: int
        @param cool_type: cooling type (linear, exponential etc.)
        @type cool_type: CoolingTypes
        @param iters: number of iterations in single temperature
        @type iters: int
        @param poly: in case of polynomial type of cooling, degree of polynomial
        @type poly: int
    """
    if cool_type == CoolingTypes.LIN:
        return linear_cooling_init(init_temp, final_temp, max_iter, iters)

    if cool_type == CoolingTypes.POLY:
        return polynomial_cooling_init(init_temp, final_temp, max_iter, iters, poly)

    if cool_type == CoolingTypes.EXP:
        return exponential_cooling_init(init_temp, final_temp, max_iter, iters)

    if cool_type == CoolingTypes.LOG:
        return logarithmic_cooling_init(init_temp, final_temp, max_iter, iters)



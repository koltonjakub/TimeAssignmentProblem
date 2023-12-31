"""Algorithm that performs simulated annealing method."""

import csv
import logging as log
import os
from inspect import signature
from typing import Callable, Any, Tuple, Union, Annotated, Type

import numpy as np
import pydantic as pdt
from pydantic import ValidationError

from tap_lib.Visualisation import Scope


class Solver(pdt.BaseModel):
    """
    Class for solving problems with the Simulated Annealing method.

    :param SolutionType: type of solution to be returned, used for validation during runtime
    :type SolutionType: Type

    :param cost: Objective function that is being optimized, takes solution as input and returns cost
    :type cost: Callable returning real value
    :param sol_gen: Function that returns solution generated randomly from the neighbourhood of the provided solution,
    takes the solution as input and returns new solution as output
    :type sol_gen: Callable returning SolutionTemplate
    :param cool: Function that cools the temperature, takes temperature and iteration as parameters
    :type cool: Callable returning non-negative value
    :param probability: Probability function, returns probability of transition to different state, takes delta_energy
    and iterations as parameters
    :type probability: Callable returning real value between 0 and 1

    :param init_sol: Starting visited_solution for the algorithm
    :type init_sol: SolutionTemplate
    :param init_temp: Starting temperature
    :type init_temp: Real value
    :param max_iterations: Max number of iterations of the main loop
    :type max_iterations: int
    :param experiment_name: name of the experiment
    :type experiment_name: str

    :param log_file_path: path of the log file
    :type log_file_path: str
    :param csv_file_path: path of the csv file
    :type csv_file_path: str
    :param log_results: flag weather to log results
    :type log_results: bool

    :param remember_iteration: flag whether to remember the vector of iterations during runtime
    :type remember_iteration: bool
    :param remember_temperature: flag whether to remember vector of temperature during runtime
    :type remember_temperature: bool
    :param remember_delta_energy: flag whether to remember vector of delta_energy during runtime
    :type remember_delta_energy: bool
    :param remember_probability_of_transition: flag whether to remember vector of probability during runtime
    :type remember_probability_of_transition: bool
    :param remember_cost_function: flag whether to remember vector of cost function during runtime
    :type remember_cost_function: bool
    :param remember_best_cost_function: flag whether to remember vector of best cost function during runtime
    :type remember_best_cost_function: bool
    :param remember_visited_solution: flag whether to remember vector of visited solution during runtime
    :type remember_visited_solution: bool
    """

    class Config(pdt.BaseConfig):
        """Config sets crucial BaseModel settings"""
        arbitrary_types_allowed = True  # Allows for validation of numpy numeric types
        validate_assignment = True  # Allows the model to validate data every time field is assigned/changed
        smart_union = True  # Prevents unnecessary casts to not matching data types
        extra = 'forbid'  # Prevents adding any unnecessary attributes to class

    # Types
    SolutionType: Type = Any

    # Callables
    cost: Callable[[Any], float] = None
    sol_gen: Callable[[Any], Any] = None
    cool: Callable[[float, int], float] = None
    probability: Callable[[float, float], float] = None

    # Initial settings
    init_sol: SolutionType = None
    init_temp: Annotated[Union[float, int], pdt.Field(gt=0)] = None
    max_iterations: Annotated[int, pdt.conint(gt=0)] = None
    experiment_name: str = pdt.Field(default="", strict=True)

    # Info-like functionalities
    log_file_path: str = None
    csv_file_path: str = None
    log_results: bool = False

    # Samples to be collected
    remember_iteration: bool = True
    remember_temperature: bool = True
    remember_delta_energy: bool = True
    remember_probability_of_transition: bool = True
    remember_cost_function: bool = True
    remember_best_cost_function: bool = True
    remember_visited_solution: bool = False

    # noinspection PyMethodParameters
    @pdt.validator('cost', pre=True, always=True)
    def validate_cost(cls, value, values):
        if value is None:
            return None

        if not callable(value):
            raise ValidationError('cost must be a callable')

        if len(signature(value).parameters) != 1:
            raise ValidationError('cost must take exactly one positional argument')

        init_sol = values.get("init_sol")
        if init_sol is None:
            return value

        result = value(init_sol)
        if not isinstance(result, (float, int)):
            raise ValidationError('cost is not a float or int for provided initial solution')

        if result < 0:
            raise ValidationError('cost is negative for provided initial solution')

        return value

    # noinspection PyMethodParameters
    @pdt.validator('sol_gen', pre=True, always=True)
    def validate_sol_gen(cls, value, values):
        if value is None:
            return value

        if not callable(value):
            raise ValidationError('sol_gen must be a callable')

        if len(signature(value).parameters) != 1:
            raise ValidationError('sol_gen must take exactly one positional argument')

        init_sol = values.get('init_sol')
        if init_sol is None:
            return value

        solution_type = values['SolutionType']
        result = value(init_sol)
        if not isinstance(result, solution_type):
            raise ValidationError(f'sol_gen must return a SolutionType: {solution_type} object')

        return value

    # noinspection PyMethodParameters
    @pdt.validator('cool', pre=True, always=True)
    def validate_cool(cls, value, values):
        if value is None:
            return value

        if not callable(value):
            raise ValidationError('cool must be a callable')

        if len(signature(value).parameters) != 2:
            raise ValidationError('cool must take exactly two positional arguments')

        init_temp = values.get("init_temp")
        if init_temp is None:
            return value

        cooled_temp = value(init_temp, 0)
        if not isinstance(cooled_temp, (float, int)):
            raise ValidationError('cool does not return int or float for provided initial temperature')

        return value

    # noinspection PyMethodParameters
    @pdt.validator('probability', pre=True, always=True)
    def validate_probability(cls, value, values):
        if value is None:
            return value

        if not callable(value):
            raise ValidationError('probability must be a callable')

        if len(signature(value).parameters) != 2:
            raise ValidationError('probability must take exactly two positional arguments')

        init_temp = values.get("init_temp")
        if init_temp is None:
            return value

        small_delta_energy = 0.25
        init_temp = values['init_temp']
        result = value(small_delta_energy, init_temp)
        if not isinstance(result, (int, float)):
            raise ValidationError('probability does not return int or float for provided initial temperature and '
                                  'delta energy: {delta}'.format(delta=small_delta_energy))

        if result < 0 or result > 1:
            raise ValidationError('probability returns value out of range(0 to 1) for provided initial temperature '
                                  'and delta energy: {delta}'.format(delta=small_delta_energy))

        return value

    # noinspection PyMethodParameters
    # noinspection PyCallingNonCallable
    @pdt.validator('init_sol', pre=True, always=True)
    def validate_init_sol(cls, value, values):
        if value is None:
            return value

        solution_type = values.get("SolutionType")
        if not isinstance(value, solution_type):
            raise ValidationError(f'init_sol must be a SolutionType: {solution_type}')

        values["init_sol"] = value
        cost = values.get("cost")
        sol_gen = values.get("sol_gen")
        cls.validate_cost(cost, values)
        cls.validate_sol_gen(sol_gen, values)

        return value

    # noinspection PyMethodParameters
    # noinspection PyCallingNonCallable
    @pdt.validator('init_temp', pre=True, always=True)
    def validate_init_temp(cls, value, values):
        if value is None:
            return value

        if not isinstance(value, (int, float)):
            raise ValidationError('init_temp must be an int or float')

        cool = values.get("cool")
        probability = values.get("probability")
        values["init_temp"] = value
        cls.validate_cool(cool, values)
        cls.validate_probability(probability, values)

        return value

    # noinspection PyMethodParameters
    @pdt.validator('max_iterations', pre=True, always=True)
    def validate_max_iterations(cls, value: int):
        if value is not None and value <= 0:
            raise pdt.ValidationError('max_iterations must be greater than 0')
        if value is not None and not isinstance(value, int):
            raise pdt.ValidationError('max_iterations must be an int')
        return value

    # noinspection PyMethodParameters
    @pdt.validator('log_file_path', pre=True, always=True)
    def validate_log_file_path(cls, value: str):
        if value is None:
            return value

        if not os.path.exists(value):
            raise FileNotFoundError('log_file_path does not exist')
        return value

    # noinspection PyMethodParameters
    @pdt.validator('csv_file_path', pre=True, always=True)
    def validate_csv_file_path(cls, value: str):
        if value is None:
            return value

        if not os.path.exists(value):
            raise pdt.ValidationError('csv_file_path does not exist')
        return value

    def setup_logger(self, log_file_path: str) -> None:
        """
        Function to set up logger for the simulated annealing runtime, logs are saved in case of an error.
        @param log_file_path: absolute path of the log file
        @type log_file_path: str
        """
        if not os.path.exists(log_file_path):
            raise FileNotFoundError('log_file_path does not exist')
        self.log_file_path = log_file_path
        log.basicConfig(filename=self.log_file_path, level=log.ERROR)

    def dump_csv(self, init_cost: Union[int, float], best_cost: Union[int, float],
                 absolute_improvement: Union[int, float], relative_improvement: float, iteration: int,
                 stopping_condition: str) -> None:
        """
        Function to dump the data to csv file
        @param init_cost: starting cost value
        @type init_cost: Union[int, float]
        @param best_cost: cost of the best solution encountered
        @type best_cost: Union[int, float]
        @param absolute_improvement: absolute value of the difference between the best and the starting solution
        @type absolute_improvement: Union[int, float], non-negative
        @param relative_improvement: ratio of improvement between the best and the starting cost
        @type relative_improvement: Union[int, float], between 0 and 1
        @param iteration: number of iterations of the main loop of the simulate annealing method
        @type iteration: int, non-negative
        @param stopping_condition: stopping criteria that terminated the algorithm
        @type stopping_condition: str
        """
        with open(self.csv_file_path, mode='a', newline='') as csv_file:
            fieldnames = ['Experiment Name', 'Initial Cost', 'Best Cost', 'Absolute Improvement',
                          'Relative Improvement', 'Iteration', 'Stopping Condition']

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if csv_file.tell() == 0:
                writer.writeheader()

            writer.writerow({
                'Experiment Name': self.experiment_name,
                'Initial Cost': init_cost,
                'Best Cost': best_cost,
                'Absolute Improvement': absolute_improvement,
                'Relative Improvement': relative_improvement,
                'Iteration': iteration,
                'Stopping Condition': stopping_condition
            })

    def simulate_annealing(self) -> Tuple[SolutionType, Scope] | None:
        """
        Function to perform simulated annealing problem for given initial conditions of certain SolutionType.
        @return: best solution encountered during runtime and scope of runtime annealing problem parameters
        @rtype: SolutionType, ScopeValid
        """

        scope: Scope = Scope()

        solution = self.init_sol
        best_solution = self.init_sol
        best_cost = self.cost(self.init_sol)
        temperature = self.init_temp

        stopping_criterion: str = 'max iterations reached'

        for it in range(0, self.max_iterations):
            neighbour = self.sol_gen(solution)

            solution_cost = self.cost(solution)
            neighbour_cost = self.cost(neighbour)

            delta_energy: float = solution_cost - neighbour_cost
            prob_of_transition: float = self.probability(delta_energy, temperature)

            if neighbour_cost < solution_cost:
                solution = neighbour
            else:
                if np.random.random(1) < prob_of_transition:
                    solution = neighbour

            temperature = self.cool(temperature, it)

            if neighbour_cost < best_cost:
                best_solution = neighbour
                best_cost = self.cost(neighbour)

            try:
                scope.iteration += [it] if self.remember_iteration else []
                scope.temperature += [temperature] if self.remember_temperature else []
                scope.probability_of_transition += [prob_of_transition] if self.remember_probability_of_transition \
                    else []
                scope.cost_function += [self.cost(solution)] if self.remember_cost_function else []
                scope.best_cost_function += [self.cost(best_solution)] if self.remember_best_cost_function else []
                scope.visited_solution += [solution] if self.remember_visited_solution else []
            except ValueError as value_error:
                log.error(f"ExpName: {self.experiment_name} resulted in TypeError: {value_error}")
                return None

        init_cost = self.cost(self.init_sol)
        absolute_improvement = init_cost - best_cost
        relative_improvement = (init_cost - best_cost) / init_cost if best_cost != 0 else 1
        iteration = np.max(scope.iteration)

        if self.log_results:
            self.dump_csv(init_cost=init_cost, best_cost=best_cost, absolute_improvement=absolute_improvement,
                          relative_improvement=relative_improvement, iteration=iteration,
                          stopping_condition=stopping_criterion)

        return best_solution, scope

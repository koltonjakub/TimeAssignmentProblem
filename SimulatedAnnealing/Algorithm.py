"""Algorithm that performs simulated annealing method."""

from typing import Callable, Any, Tuple
from numpy.random import random
from pydantic import ValidationError
from SimulatedAnnealing.Visualisation.Visualisation import UnvalidatedScope, ValidatedScope

import logging as log
import numpy as np
import time


class SimulatedAnnealingValidationError(Exception):
    def __init__(self, message, value=None) -> None:
        self.value = value
        self.message = message
        super().__init__(self.message)


def generate_sa_algorithm(SolutionTemplate: Any) -> Callable:
    # TODO move param validation and setup_logger here
    def setup_logger(log_file: str) -> None:
        log.basicConfig(filename=log_file, level=log.INFO)

    def simulated_annealing(cost: Callable[[SolutionTemplate], float],
                            init_sol: SolutionTemplate,
                            sol_gen: Callable[[SolutionTemplate], SolutionTemplate],
                            init_temp: float,
                            cool: Callable[[float, int], float],
                            probability: Callable[[float, float], float],
                            max_iterations: int = 1000,
                            log_file_path: str = '../logs/simulated_annealing.log',
                            experiment_name: str = 'SimulatedAnnealing') -> Tuple[SolutionTemplate, ValidatedScope]:
        """
        Function takes all the arguments that simulated annealing method requires and returns the best possible
        minimal visited_solution.

        :param cost: Objective function that is being optimized
        :type cost: Callable returning real value
        :param init_sol: Starting visited_solution for the algorithm
        :type init_sol: SolutionTemplate
        :param sol_gen: Function that returns next visited_solution, randomly different from provided
        :type sol_gen: Callable returning SolutionTemplate
        :param init_temp: Starting temperature
        :type init_temp: Real value
        :param cool: Function that cools the temperature
        :type cool: Callable returning non-negative value
        :param probability: Probability function, returns probability of transition to different from delta energy
        :type probability: Callable returning real value between 0 and 1
        :param max_iterations: Max iterations of the main loop
        :type max_iterations: int
        :param log_file_path: path of the log file
        :type log_file_path: str
        :param experiment_name: name of the experiment
        :type experiment_name: str
        :return: Most optimal visited_solution that has been found during search, runtime simulation values
        :rtype: SolutionTemplate, UnvalidatedScope
        """

        # TODO add validation for SolutionTemplate Type, can be performed within solution generator function
        # TODO add remembering of the best encountered solution

        # TODO implement logger

        if not callable(cost):
            raise SimulatedAnnealingValidationError(message="objective_value is not callable")

        if not callable(sol_gen):
            raise SimulatedAnnealingValidationError(message="sol_gen is not callable")

        if not isinstance(init_temp, (int, float, np.int32, np.int64, np.float32, np.float64)):
            raise SimulatedAnnealingValidationError(message="init_temp must be int or float or np.int32 or np.int64 or "
                                                            "np.float32 or np.float64")

        if not callable(cool):
            raise SimulatedAnnealingValidationError(message="cool is not callable")

        if not callable(probability):
            raise SimulatedAnnealingValidationError(message="probability is not callable")

        if not isinstance(max_iterations, (int, np.int32, np.int64)):
            raise SimulatedAnnealingValidationError(message='max_iterations must be int', value=max_iterations)

        setup_logger(log_file=log_file_path)

        invalid_simul_scope: UnvalidatedScope = UnvalidatedScope()

        solution: SolutionTemplate = init_sol
        best_solution: SolutionTemplate = init_sol
        best_cost = cost(init_sol)
        temperature: float = init_temp

        stopping_criterion: str = 'max iterations reached'

        for it in range(0, max_iterations):
            neighbour: SolutionTemplate = sol_gen(solution)

            solution_cost = cost(solution)
            neighbour_cost = cost(neighbour)

            delta_energy: float = solution_cost - neighbour_cost
            prob_of_transition: float = probability(delta_energy, temperature)

            if neighbour_cost < solution_cost:
                solution: SolutionTemplate = neighbour
            else:
                if random(1) < prob_of_transition:
                    solution: SolutionTemplate = neighbour

            if neighbour_cost < best_cost:
                best_solution: SolutionTemplate = neighbour
                best_cost = cost(neighbour)

            temperature: float = cool(temperature, it)

            invalid_simul_scope.iteration += [it]
            invalid_simul_scope.temperature += [temperature]
            invalid_simul_scope.probability_of_transition += [prob_of_transition]
            invalid_simul_scope.cost_function += [cost(solution)]
            invalid_simul_scope.best_cost_function += [cost(best_solution)]
            invalid_simul_scope.visited_solution += [solution]

        simul_scope: ValidatedScope = ValidatedScope()
        try:
            simul_scope.iteration = invalid_simul_scope.iteration
            simul_scope.temperature = invalid_simul_scope.temperature
            simul_scope.probability_of_transition = invalid_simul_scope.probability_of_transition
            simul_scope.cost_function = invalid_simul_scope.cost_function
            simul_scope.best_cost_function = invalid_simul_scope.best_cost_function
            simul_scope.visited_solution = invalid_simul_scope.visited_solution
        except ValidationError as e:
            log.error(f"ExpName: {experiment_name} resulted in Pydantic validation error: {e}")
            for error in e.errors():
                log.error(f"Field: {error['loc']}, Error: {error['msg']}")

        init_cost = cost(init_sol)
        absolute_improvement = init_cost - best_cost
        relative_improvement = (init_cost - best_cost) / init_cost * 100 if best_cost != 0 else 100

        log.info(f"|Exp Name: {experiment_name} | {stopping_criterion} in {np.max(simul_scope.iteration)} iterations |\n"
                 f"| abs_improvement = {absolute_improvement} | rel_improvement = {relative_improvement} | "
                 f"cost = {best_cost} |")

        return best_solution, simul_scope

    return simulated_annealing


def main():
    from typing import Tuple, Union
    from numpy.random import randint
    from numpy import exp
    from Visualisation.Visualisation import plot_scope
    import matplotlib.pyplot as plt

    def cost_fun(vec):
        (x, y) = vec

        return x ** 2 + y ** 2

    def sol_gen_fun(vec):
        (x, y) = vec
        step = 0.01

        if randint(2) == 0:
            if randint(2) == 0:
                x += step
            else:
                x -= step
        else:
            if randint(2) == 0:
                y += step
            else:
                y -= step

        if x < -5:
            x = -5
        if x > 5:
            x = 5
        if y < -5:
            y = -5
        if y > 5:
            y = 5

        return x, y

    def colling_fun(_, k):
        temp = (1 - (k + 1) / max_iter) * start_temp

        if temp <= 10 ** (-9):
            return 0.0001
        return temp

    def prob_fun(delta_en, temp):
        if delta_en < 0:
            return 1
        return exp(-delta_en / (1.380649 * temp))

    solution_type = Tuple[Union[int, float], Union[int, float]]
    sim_an = generate_sa_algorithm(solution_type)

    max_iter = 10 ** 3
    start_temp = 0.1
    init_sol_tup = (2, -2)

    sol, scope = sim_an(cost=cost_fun,
                        init_sol=init_sol_tup,
                        sol_gen=sol_gen_fun,
                        init_temp=start_temp,
                        cool=colling_fun,
                        probability=prob_fun,
                        max_iterations=max_iter)

    plot_scope(scope)
    print(sol)

    x_vec = [tup[0] for tup in scope.visited_solution]
    y_vec = [tup[1] for tup in scope.visited_solution]

    plt.plot(x_vec, y_vec)
    plt.plot(0, 0, 'r.')
    plt.plot(sol[0], sol[1], 'g.')
    plt.title('solutions')
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    main()
    stop_time = time.time()
    print('Execution time: ' + str(stop_time - start_time))

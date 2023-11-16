"""Algorithm that performs simulated annealing method."""

from typing import Callable, Any, Tuple
from numpy import int32, int64, float32, float64
from numpy.random import random
from pydantic import ValidationError
import logging as log
import time
from SimulatedAnnealing.Visualisation.Visualisation import UnvalidatedScope, ValidatedScope


class SimulatedAnnealingValidationError(Exception):
    def __init__(self, message, value=None) -> None:
        self.value = value
        self.message = message
        super().__init__(self.message)


def generate_sa_algorithm(SolutionTemplate: Any) -> Callable:
    def simulated_annealing(cost: Callable[[SolutionTemplate], float],
                            init_sol: SolutionTemplate,
                            sol_gen: Callable[[SolutionTemplate], SolutionTemplate],
                            init_temp: float,
                            cool: Callable[[float, int], float],
                            probability: Callable[[float, float], float],
                            max_iterations: int = 1000) -> Tuple[SolutionTemplate, ValidatedScope]:
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
        :return: Most optimal visited_solution that has been found during search, runtime simulation values
        :rtype: SolutionTemplate, UnvalidatedScope
        """

        # TODO add validation for SolutionTemplate Type, can be performed within solution generator function
        # TODO add remembering of the best encountered solution

        if not callable(cost):
            raise SimulatedAnnealingValidationError(message="objective_value is not callable")

        if not callable(sol_gen):
            raise SimulatedAnnealingValidationError(message="sol_gen is not callable")

        if not isinstance(init_temp, (int, float, int32, int64, float32, float64)):
            raise SimulatedAnnealingValidationError(message="init_temp must be int or float or int32 or int64 or "
                                                            "float32 or float64")

        if not callable(cool):
            raise SimulatedAnnealingValidationError(message="cool is not callable")

        if not callable(probability):
            raise SimulatedAnnealingValidationError(message="probability is not callable")

        if not isinstance(max_iterations, (int, int32, int64)):
            raise SimulatedAnnealingValidationError(message='max_iterations must be int', value=max_iterations)

        invalid_simul_scope: UnvalidatedScope = UnvalidatedScope()

        solution: SolutionTemplate = init_sol
        best_solution: SolutionTemplate = init_sol
        best_cost = cost(init_sol)
        temperature: float = init_temp

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

            invalid_simul_scope.iteration += [it]
            invalid_simul_scope.temperature += [temperature]
            invalid_simul_scope.probability_of_transition += [prob_of_transition]
            invalid_simul_scope.cost_function += [cost(solution)]
            invalid_simul_scope.best_cost_function += [cost(best_solution)]
            invalid_simul_scope.visited_solution += [solution]

            temperature: float = cool(temperature, it)

        try:
            simul_scope: ValidatedScope = ValidatedScope()
            simul_scope.iteration = invalid_simul_scope.iteration
            simul_scope.temperature = invalid_simul_scope.temperature
            simul_scope.probability_of_transition = invalid_simul_scope.probability_of_transition
            simul_scope.cost_function = invalid_simul_scope.cost_function
            simul_scope.best_cost_function = invalid_simul_scope.best_cost_function
            simul_scope.visited_solution = invalid_simul_scope.visited_solution
        except ValidationError as e:
            print(e)
        else:
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

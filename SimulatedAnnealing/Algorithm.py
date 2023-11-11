"""Algorithm that performs simulated annealing method."""

from typing import Callable, Any, Dict, List, Tuple
from numpy import int32, int64, float32, float64
from numpy.random import random
from SimulatedAnnealing.Utils.DataTypes import ScopeParams, Scope, PositiveNumber, RealNumber, NonNegativeInt


class SimulatedAnnealingValidationError(Exception):
    def __init__(self, message, value=None) -> Exception:
        self.value = value
        self.message = message
        super().__init__(self.message)


def generate_sa_algorithm(SolutionTemplateType: Any) -> Callable:
    def simulated_annealing(cost: Callable[[SolutionTemplateType], RealNumber],
                            init_sol: SolutionTemplateType,
                            sol_gen: Callable[[SolutionTemplateType], SolutionTemplateType],
                            init_temp: PositiveNumber,
                            cool: Callable[[PositiveNumber, int], PositiveNumber],
                            probability: Callable[[RealNumber, PositiveNumber], RealNumber],
                            max_iterations: NonNegativeInt = 1000) -> Tuple[SolutionTemplateType, Scope]:
        """
        Function takes all the arguments that simulated annealing method requires and returns the best possible
        minimal visited_solution.

        :param cost: Objective function that is being optimized
        :type cost: Callable returning real value
        :param init_sol: Starting visited_solution for the algorithm
        :type init_sol: SolutionTemplateType
        :param sol_gen: Function that returns next visited_solution, randomly different from provided
        :type sol_gen: Callable returning SolutionTemplateType
        :param init_temp: Starting temperature
        :type init_temp: Real value
        :param cool: Function that cools the temperature
        :type cool: Callable returning non-negative value
        :param probability: Probability function, returns probability of transition to different from delta energy
        :type probability: Callable returning real value between 0 and 1
        :param max_iterations: Max iterations of the main loop
        :type max_iterations: int
        :return: Most optimal visited_solution that has been found during search, runtime simulation values
        :rtype: SolutionTemplateType, Scope
        """

        # TODO add validation for SolutionTemplateType Type, can be performed within solution generator function

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

        simulation_scope_data: Dict[ScopeParams.name: List] = {
            ScopeParams.iteration: list(),
            ScopeParams.temperature: list(),
            ScopeParams.delta_energy: list(),
            ScopeParams.probability_of_transition: list(),
            ScopeParams.cost_function: list(),
            ScopeParams.visited_solution: list()
        }

        def update_scope_data(scope_data, i, temp, delta_en, prob, cost_val, sol):
            scope_data[ScopeParams.iteration].append(i)
            scope_data[ScopeParams.temperature].append(temp)
            scope_data[ScopeParams.delta_energy].append(delta_en)
            scope_data[ScopeParams.probability_of_transition].append(prob)
            scope_data[ScopeParams.cost_function].append(cost_val)
            scope_data[ScopeParams.visited_solution].append(sol)

        solution: SolutionTemplateType = init_sol
        temperature: PositiveNumber = init_temp

        for it in range(0, max_iterations):
            neighbour: SolutionTemplateType = sol_gen(solution)

            solution_cost = cost(solution)
            neighbour_cost = cost(neighbour)

            delta_energy: RealNumber = solution_cost - neighbour_cost
            prob_of_transition: RealNumber = probability(delta_energy, temperature)

            if neighbour_cost < solution_cost:
                solution: SolutionTemplateType = neighbour
            else:
                if random(1) < prob_of_transition:
                    solution: SolutionTemplateType = neighbour

            update_scope_data(scope_data=simulation_scope_data,
                              i=it,
                              temp=temperature,
                              delta_en=delta_energy,
                              prob=prob_of_transition,
                              cost_val=cost(solution),
                              sol=solution)

            temperature: PositiveNumber = cool(temperature, it)

        # create Scope only after the algorithm has ended, validations take too long if validation of reassignments
        # were to be performed in the main loop
        simulation_scope = Scope()

        simulation_scope.iteration = simulation_scope_data[ScopeParams.iteration]
        simulation_scope.temperature = simulation_scope_data[ScopeParams.temperature]
        simulation_scope.delta_energy = simulation_scope_data[ScopeParams.delta_energy]
        simulation_scope.probability_of_transition = simulation_scope_data[ScopeParams.probability_of_transition]
        simulation_scope.cost_function = simulation_scope_data[ScopeParams.cost_function]
        simulation_scope.visited_solution = simulation_scope_data[ScopeParams.visited_solution]

        return solution, simulation_scope

    return simulated_annealing


def main():
    from typing import Tuple, Union
    from numpy.random import randint
    from numpy import exp
    from SimulatedAnnealing.Visualisation import plot_scope
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
    main()

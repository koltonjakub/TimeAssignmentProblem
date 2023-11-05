"""This file contains all visual functionalities"""


from matplotlib.pyplot import subplots, show
from numpy import min, max
from SimulatedAnnealing.FactoryAssignmentProblem.DataTypes import Scope, ScopeParams


def plot_scope(scope: Scope) -> None:
    """
    Function that plots runtime values of simulation from Scope type.

    :param scope: Scope of simulated annealing algorithm
    :type scope: Scope
    :return: None    :rtype:
    """

    figure, axes = subplots(2, 2, figsize=(10, 10))

    axes[0, 0].plot(scope.iteration, scope.temperature)
    axes[0, 0].set_title(ScopeParams.temperature.value)
    axes[0, 0].set_xlabel(ScopeParams.iteration.value)
    axes[0, 0].set_ylabel(ScopeParams.temperature.value)
    axes[0, 0].grid(True)

    axes[1, 0].plot(scope.iteration, scope.delta_energy, 'b.', markersize=1)
    axes[1, 0].set_title(ScopeParams.delta_energy.value)
    axes[1, 0].set_xlabel(ScopeParams.iteration.value)
    axes[1, 0].set_ylabel(ScopeParams.delta_energy.value)
    axes[1, 0].grid(True)

    axes[1, 1].plot(scope.iteration, scope.probability_of_transition, 'b.', markersize=1)
    axes[1, 1].set_title(ScopeParams.probability_of_transition.value)
    axes[1, 1].set_xlabel(ScopeParams.iteration.value)
    axes[1, 1].set_ylabel(ScopeParams.probability_of_transition.value)
    axes[1, 1].grid(True)

    axes[0, 1].plot(scope.iteration, scope.cost_function)
    axes[0, 1].set_title(ScopeParams.cost_function.value)
    axes[0, 1].set_xlabel(ScopeParams.iteration.value)
    axes[0, 1].set_ylabel(ScopeParams.cost_function.value)
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim([min([0, min(scope.cost_function)]), max(scope.cost_function) + 0.1])

    show()

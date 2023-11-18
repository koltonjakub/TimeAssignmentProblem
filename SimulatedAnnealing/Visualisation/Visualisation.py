"""This file contains all visual functionalities"""


from matplotlib.pyplot import subplots, show
from numpy import min, max
from pydantic import BaseModel, BaseConfig, Field, conint, confloat
from typing import List, Any, Dict


class ScopeNoValid(BaseModel):
    """BaseModel that stores all runtime data of simulation"""

    class Config(BaseConfig):
        """Config sets crucial BaseModel settings"""
        arbitrary_types_allowed = True  # Allows for validation of numpy numeric types
        validate_assignment = False  # Allows the model to validate data every time field is assigned/changed
        smart_union = True  # Prevents unnecessary casts to not matching data types

    iteration: List[conint(ge=0, strict=True)] = []
    temperature: List[confloat(ge=0)] = []
    delta_energy: List[confloat()] = []
    probability_of_transition: List[confloat(ge=0, le=1)] = []
    cost_function: List[confloat(ge=0)] = []
    best_cost_function: List[confloat(ge=0)] = []
    visited_solution: List[Any] = []
    label: Dict[str, str] = Field({
        'iteration': 'encounter',
        'temperature': 'temperature',
        'delta_energy': "delta_energy",
        'probability_of_transition': "prob_of_trans",
        'cost_function': 'objective_value',
        'best_cost_function': 'best_cost_function',
        'visited_solution': 'visited_solution'
    }, frozen=True)

    # TODO consider writing custom @validator that checks only one value rather than whole list when new
    # TODO value is assigned


class ScopeValid(ScopeNoValid):
    class Config(BaseConfig):
        """Config sets crucial BaseModel settings"""
        arbitrary_types_allowed = True  # Allows for validation of numpy numeric types
        validate_assignment = True  # Allows the model to validate data every time field is assigned/changed
        smart_union = True  # Prevents unnecessary casts to not matching data types


def plot_scope(scope: ScopeNoValid) -> None:
    """
    Function that plots runtime values of simulation from Scope type.
    @param scope: Scope of simulated annealing algorithm runtime values
    @type scope: ScopeNoValid
    """

    figure, axes = subplots(2, 2, figsize=(10, 10))

    axes[0, 0].plot(scope.iteration, scope.temperature)
    axes[0, 0].set_title(scope.label['temperature'])
    axes[0, 0].set_xlabel(scope.label['iteration'])
    axes[0, 0].set_ylabel(scope.label['temperature'])
    axes[0, 0].grid(True)

    axes[0, 1].plot(scope.iteration, scope.probability_of_transition, 'b.', markersize=1)
    axes[0, 1].set_title(scope.label['probability_of_transition'])
    axes[0, 1].set_xlabel(scope.label['iteration'])
    axes[0, 1].set_ylabel(scope.label['probability_of_transition'])
    axes[0, 1].grid(True)

    axes[1, 0].plot(scope.iteration, scope.best_cost_function)
    axes[1, 0].set_title(scope.label['best_cost_function'])
    axes[1, 0].set_xlabel(scope.label['iteration'])
    axes[1, 0].set_ylabel(scope.label['best_cost_function'])
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim([min([0, min(scope.best_cost_function)]), max(scope.best_cost_function) + 0.1])

    axes[1, 1].plot(scope.iteration, scope.cost_function)
    axes[1, 1].set_title(scope.label['cost_function'])
    axes[1, 1].set_xlabel(scope.label['iteration'])
    axes[1, 1].set_ylabel(scope.label['cost_function'])
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim([min([0, min(scope.cost_function)]), max(scope.cost_function) + 0.1])

    show()


if __name__ == "__main__":
    def scope_usage_example():
        scp = ScopeValid()
        scp.iteration = [1, 0.1]  # Type coercion is always ON, probably due to
        scp.iteration += [1]  # The only proper way to append to list, .append() method omits the pydantic validation

        scp.delta_energy = [-1]
        scp.delta_energy += [-1]

        scp.visited_solution += [(1, 1)]

        print('ScopeNoValid:')
        print(scp)
        print()

    scope_usage_example()

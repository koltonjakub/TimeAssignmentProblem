"""This file contains all visual functionalities"""


from matplotlib.pyplot import subplots, show
from numpy import min, max
from pydantic import BaseModel, BaseConfig, Field, conint, confloat
from typing import List, Any, Annotated
from enum import Enum


class UnvalidatedScope(BaseModel):
    """BaseModel that stores all runtime data of simulation"""

    class Config(BaseConfig):
        """Config sets crucial BaseModel settings"""
        arbitrary_types_allowed = True  # Allows for validation of numpy numeric types
        validate_assignment = False  # Allows the model to validate data every time field is assigned/changed
        smart_union = True  # Prevents unnecessary casts to not matching data types

    # TODO add best solution encountered cost function field, chatGPT suggests:
    # 1: Class Variables
    # 2: extra field: Dict[str: str] - maps attribute name to its label
    # 3: Custom Metaclass or Decorator(probably overkill)

    iteration: List[conint(ge=0, strict=True)] = []
    temperature: List[confloat(ge=0)] = []
    delta_energy: List[confloat()] = []  # probably useless
    probability_of_transition: List[confloat(ge=0, le=1)] = []
    cost_function: List[confloat(ge=0)] = []
    best_cost_function: List[confloat(ge=0)] = []
    visited_solution: List[Any] = []

    # TODO consider writing custom @validator that checks only one value rather than whole list when new
    # TODO value is assigned


class ValidatedScope(UnvalidatedScope):
    class Config(BaseConfig):
        """Config sets crucial BaseModel settings"""
        arbitrary_types_allowed = True  # Allows for validation of numpy numeric types
        validate_assignment = True  # Allows the model to validate data every time field is assigned/changed
        smart_union = True  # Prevents unnecessary casts to not matching data types


class ScopeParams(Enum):
    """Enum type of all data available within UnvalidatedScope type"""

    iteration: str = 'encounter'
    temperature: str = 'temperature'
    delta_energy: str = "delta_energy"
    probability_of_transition: str = "prob_of_trans"
    cost_function: str = 'objective_value'
    best_cost_function: str = 'best_cost_function'
    visited_solution: str = 'visited_solution'


def plot_scope(scope: UnvalidatedScope) -> None:
    """
    Function that plots runtime values of simulation from Scope type.
    @param scope: Scope of simulated annealing algorithm runtime values
    @type scope: UnvalidatedScope
    """

    figure, axes = subplots(2, 2, figsize=(10, 10))

    axes[0, 0].plot(scope.iteration, scope.temperature)
    axes[0, 0].set_title(ScopeParams.temperature.value)
    axes[0, 0].set_xlabel(ScopeParams.iteration.value)
    axes[0, 0].set_ylabel(ScopeParams.temperature.value)
    axes[0, 0].grid(True)

    axes[0, 1].plot(scope.iteration, scope.probability_of_transition, 'b.', markersize=1)
    axes[0, 1].set_title(ScopeParams.probability_of_transition.value)
    axes[0, 1].set_xlabel(ScopeParams.iteration.value)
    axes[0, 1].set_ylabel(ScopeParams.probability_of_transition.value)
    axes[0, 1].grid(True)

    axes[1, 0].plot(scope.iteration, scope.best_cost_function)
    axes[1, 0].set_title(ScopeParams.best_cost_function.value)
    axes[1, 0].set_xlabel(ScopeParams.iteration.value)
    axes[1, 0].set_ylabel(ScopeParams.best_cost_function.value)
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim([min([0, min(scope.best_cost_function)]), max(scope.best_cost_function) + 0.1])

    axes[1, 1].plot(scope.iteration, scope.cost_function)
    axes[1, 1].set_title(ScopeParams.cost_function.value)
    axes[1, 1].set_xlabel(ScopeParams.iteration.value)
    axes[1, 1].set_ylabel(ScopeParams.cost_function.value)
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim([min([0, min(scope.cost_function)]), max(scope.cost_function) + 0.1])

    show()


if __name__ == "__main__":
    def scope_usage_example():
        scp = UnvalidatedScope()
        scp.iteration = [1, 0, 2.0]  # Type coercion is always ON, probably due to
        scp.iteration += [1]  # The only proper way to append to list, .append() method omits the pydantic validation

        scp.probability_of_transition = [1.2]
        scp.probability_of_transition += [1.2]
        scp.delta_energy = [-1]
        scp.delta_energy += [-1]

        scp.visited_solution += [(1, 1)]

        print('UnvalidatedScope:')
        print(scp)
        print()

    scope_usage_example()

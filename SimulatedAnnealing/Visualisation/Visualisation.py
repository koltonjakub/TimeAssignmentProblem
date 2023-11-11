"""This file contains all visual functionalities"""


from matplotlib.pyplot import subplots, show
from numpy import min, max, int32, int64, float32, float64
from pydantic import BaseModel, BaseConfig, Field
from typing import List, Union, Any, Annotated
from enum import Enum


NonNegativeInt = Union[int, int32, int64]
RealNumber = Union[float, float32, float64, int, int32, int64]
PositiveNumber = Union[float, float32, float64, int, int32, int64]


class Scope(BaseModel):
    """BaseModel that stores all runtime data of simulation"""

    class Config(BaseConfig):
        """Config sets crucial BaseModel settings"""
        arbitrary_types_allowed = True  # Allows for validation of numpy numeric types
        validate_assignment = True  # Allows the model to validate data every time_span field is assigned/changed
        smart_union = True  # Prevents unnecessary rounding to int

    # TODO add best solution encountered cost function field
    iteration: Annotated[List[NonNegativeInt], Field(ge=0)] = []
    temperature: Annotated[List[PositiveNumber], Field(ge=0)] = []
    delta_energy: Annotated[List[RealNumber], Field()] = []
    probability_of_transition: Annotated[List[RealNumber], Field(ge=0, le=1)] = []
    cost_function: Annotated[List[RealNumber], Field()] = []
    visited_solution: Annotated[List[Any], Any] = []


class ScopeParams(Enum):
    """Enum type of all data available within Scope type"""

    # TODO add best solution encountered cost function param
    iteration: str = 'encounter'
    temperature: str = 'temperature'
    delta_energy: str = "delta_energy"
    probability_of_transition: str = "prob_of_trans"
    cost_function: str = 'objective_value'
    visited_solution: str = 'visited_solution'


def plot_scope(scope: Scope) -> None:
    """
    Function that plots runtime values of simulation from Scope type.
    @param scope: Scope of simulated annealing algorithm
    @type scope: Scope
    """

    # TODO rearrange subplots:
    # temp                 prob_of_trans
    # best_objective_value objective_value

    figure, axes = subplots(2, 2, figsize=(10, 10))

    axes[0, 0].plot(scope.iteration, scope.temperature)
    axes[0, 0].set_title(ScopeParams.temperature.value)
    axes[0, 0].set_xlabel(ScopeParams.iteration.value)
    axes[0, 0].set_ylabel(ScopeParams.temperature.value)
    axes[0, 0].grid(True)

    # TODO change delta_energy plot to best function cost_encountered
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


if __name__ == "__main__":
    def scope_usage_example():
        scp = Scope()
        scp.iteration = [1, 0, 2.0]  # Type coercion is always ON, probably due to
        scp.iteration += [1]  # The only proper way to append to list, .append() method omits the pydantic validation

        scp.probability_of_transition = [1.2]
        scp.probability_of_transition += [1.2]
        scp.delta_energy = [-1]
        scp.delta_energy += [-1]

        scp.visited_solution += [(1, 1)]

        print('Scope:')
        print(scp)
        print()

    scope_usage_example()

"""This file contains all visual functionalities"""

from pydantic import BaseModel, BaseConfig, Field, conint, confloat
from dataclasses import dataclass, field
from typing import List, Any, Dict, Union
from matplotlib.pyplot import subplots, show
from numpy import min, max


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


class ScopeValid(ScopeNoValid):
    class Config(BaseConfig):
        """Config sets crucial BaseModel settings"""
        arbitrary_types_allowed = True  # Allows for validation of numpy numeric types
        validate_assignment = True  # Allows the model to validate data every time field is assigned/changed
        smart_union = True  # Prevents unnecessary casts to not matching data types


# class Scope(BaseModel):
#     """BaseModel that stores all runtime data of simulation"""
#
#     class Config(BaseConfig):
#         """Config sets crucial BaseModel settings"""
#         arbitrary_types_allowed = True  # Allows for validation of numpy numeric types
#         validate_assignment = True  # Allows the model to validate data every time field is assigned/changed
#         smart_union = True  # Prevents unnecessary casts to not matching data types
#         extra = 'forbid'
#
#     iteration: List[conint(ge=0, strict=True)] = []
#     temperature: List[confloat(ge=0)] = []
#     delta_energy: List[confloat()] = []
#     probability_of_transition: List[confloat(ge=0, le=1)] = []
#     cost_function: List[confloat(ge=0)] = []
#     best_cost_function: List[confloat(ge=0)] = []
#     visited_solution: List[Any] = []
#     label: Annotated[Dict[str, str], Field(frozen=True)] = {
#         'iteration': 'iteration',
#         'temperature': 'temperature',
#         'delta_energy': "delta_energy",
#         'probability_of_transition': "probability",
#         'cost_function': 'objective_value',
#         'best_cost_function': 'best_objective_value',
#         'visited_solution': 'visited_solution'
#     }


@dataclass
class Scope:
    """Class that stores all runtime data of simulation"""
    iteration: List[int] = field(default_factory=list)
    temperature: List[Union[float, int]] = field(default_factory=list)
    delta_energy: List[Union[float, int]] = field(default_factory=list)
    probability_of_transition: List[float] = field(default_factory=list)
    cost_function: List[Union[int, float]] = field(default_factory=list)
    best_cost_function: List[Union[int, float]] = field(default_factory=list)
    visited_solution: List[Any] = field(default_factory=list)
    label: Dict[str, str] = field(default_factory=lambda: {
        'iteration': 'iteration',
        'temperature': 'temperature',
        'delta_energy': "delta_energy",
        'probability_of_transition': "probability",
        'cost_function': 'objective_value',
        'best_cost_function': 'best_objective_value',
        'visited_solution': 'visited_solution'
    })

    def __post_init__(self):
        for field_name, field_value in self.__dict__.items():
            type(self).validate_field(field_name, field_value)

    def __setattr__(self, name, value):
        label = getattr(self, 'label', None)
        if label is not None and name in label and len(value) > 0:
            if not isinstance(value, List):
                raise ValueError(f'Value {name} must be iterable')
            type(self).validate_field(name, [value[-1]])  # validate only the last element

        super().__setattr__(name, value)

    @classmethod
    def validate_field(cls, field_name, field_value) -> None:
        cls.validate_iteration(field_name, field_value)
        cls.validate_temperature(field_name, field_value)
        cls.validate_delta_energy(field_name, field_value)
        cls.validate_probability(field_name, field_value)
        cls.validate_cost_function(field_name, field_value)
        cls.validate_best_cost_function(field_name, field_value)

    @staticmethod
    def validate_iteration(field_name: str, field_value: List[Union[int]]) -> None:
        if field_name == "iteration" and not all(isinstance(value, int) and value >= 0 for value in field_value):
            raise ValueError(f"All values in {field_name} must be non-negative int")

    @staticmethod
    def validate_temperature(field_name: str, field_value: List[Union[int, float]]) -> None:
        if (field_name == "temperature" and
                not all(isinstance(value, (int, float)) and value >= 0 for value in field_value)):
            raise ValueError(f"All values in {field_name} must be non-negative int or float")

    @staticmethod
    def validate_delta_energy(field_name: str, field_value: List[Union[int, float]]) -> None:
        if field_name == "delta_energy" and not all(isinstance(value, (int, float)) for value in field_value):
            raise ValueError(f"All values in {field_name} must be int or float")

    @staticmethod
    def validate_probability(field_name: str, field_value) -> None:
        if (field_name == "probability_of_transition" and
                not all(isinstance(value, (float, int)) and 0 <= value <= 1 for value in field_value)):
            raise ValueError(f"Values in {field_name} must be between 0 and 1, int or float")

    @staticmethod
    def validate_cost_function(field_name: str, field_value) -> None:
        if (field_name == "cost_function" and
                not all(isinstance(value, (float, int)) and 0 <= value for value in field_value)):
            raise ValueError(f"Values in {field_name} must be between non-negative int or float")

    @staticmethod
    def validate_best_cost_function(field_name: str, field_value) -> None:
        if (field_name == "best_cost_function" and
                not all(isinstance(value, (float, int)) and 0 <= value for value in field_value)):
            raise ValueError(f"Values in {field_name} must be between non-negative int or float")


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

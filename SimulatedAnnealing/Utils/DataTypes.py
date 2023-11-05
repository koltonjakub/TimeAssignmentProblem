"""This file contains all data types used in project"""

from numpy import max, int32, int64, float32, float64
from pydantic import BaseModel, Field, BaseConfig, conint
from typing import Union, Dict, List, Any, Annotated
from collections import defaultdict
from enum import Enum


NonNegativeInt = Union[int, int32, int64]
RealNumber = Union[float, float32, float64, int, int32, int64]
PositiveNumber = Union[float, float32, float64, int, int32, int64]


class Scope(BaseModel):
    """BaseModel that stores all the runtime data of simulation"""

    class Config(BaseConfig):
        """Config sets crucial BaseModel settings"""
        arbitrary_types_allowed = True  # Allows for validation of numpy numeric types
        validate_assignment = True  # Allows the model to validate data every time field is assigned/changed
        smart_union = True  # Prevents unnecessary rounding to int

    iteration: Annotated[List[NonNegativeInt], Field(ge=0)] = []
    temperature: Annotated[List[PositiveNumber], Field(ge=0)] = []
    delta_energy: Annotated[List[RealNumber], Field()] = []
    probability_of_transition: Annotated[List[RealNumber], Field(ge=0, le=1)] = []
    cost_function: Annotated[List[RealNumber], Field()] = []
    visited_solution: Annotated[List[Any], Any] = []


class ScopeParams(Enum):
    """Enum type of all data available within Scope type"""

    iteration: str = 'iteration'
    temperature: str = 'temperature'
    delta_energy: str = "delta_energy"
    probability_of_transition: str = "prob_of_trans"
    cost_function: str = 'cost'
    visited_solution: str = 'visited_solution'


class Resource(BaseModel):
    """Informal interface for any type of resource_type that may be used in a factory"""

    hourly_cost: Union[int, float]
    hourly_gain: Union[int, float]

    id: conint(ge=0)


class Employee(Resource):
    """Class representing an employee in factory"""

    name: str
    surname: str

    # TODO add the rest of necessary fields


class Machine(Resource):
    """Class representing a machine in factory"""

    inventory_nr: str

    # TODO add the rest of necessary fields


class AvailableResources(Enum):
    """Enum type, lists all available types of resources"""
    EMPLOYEE = Employee
    MACHINE = Machine


class ResourceManager:
    """Class that manages all types of resources within the project"""

    def __init__(self):
        self.used_ids: Dict[AvailableResources, int] = defaultdict(None)

    def create_resource(self, data, resource_type: AvailableResources) -> Resource:
        data['id']: int = self.update_resource_ids(resource_type=resource_type)

        # TODO add try statement, if resource is not created, delete last inserted id
        resource: Resource = self.map_resource_class(data=data, resource_type=resource_type)

        return resource

    def update_resource_ids(self, resource_type: AvailableResources) -> int:
        if resource_type not in self.used_ids:  # assign first id of provided Resource type
            new_id = 0
            self.used_ids[resource_type]: List[int] = [new_id]

            return new_id

        else:  # increment id of provided type
            new_id = max(self.used_ids[resource_type]) + 1
            self.used_ids[resource_type].append(new_id)

            return new_id

    @classmethod
    def map_resource_class(cls, data, resource_type: AvailableResources) -> Resource:
        if resource_type == AvailableResources.MACHINE:
            return Machine(**data)

        if resource_type == AvailableResources.EMPLOYEE:
            return Employee(**data)


def resources_creation_example():
    manager = ResourceManager()  # call the manager

    employees = [manager.create_resource(data={'hourly_cost': 1, 'hourly_gain': 1,  # pass data
                                               'name': 'John', 'surname': 'Smith'},  # pass data
                                         resource_type=AvailableResources.EMPLOYEE),  # pass type
                 manager.create_resource(data={'hourly_cost': 1, 'hourly_gain': 1,
                                               'name': 'Adam', 'surname': 'Nash'},
                                         resource_type=AvailableResources.EMPLOYEE)]

    machines = [manager.create_resource(data={'hourly_cost': 1, 'hourly_gain': 1, 'inventory_nr': 11},
                                        resource_type=AvailableResources.MACHINE),
                manager.create_resource(data={'hourly_cost': 1, 'hourly_gain': 1, 'inventory_nr': 12},
                                        resource_type=AvailableResources.MACHINE)]

    print('Generated employees:')
    for empl in employees:
        print(empl)

    print()

    print('Generated machines:')
    for mach in machines:
        print(mach)

    print()


def validation_errors_example():
    manager = ResourceManager()

    # proper definition od machine instance
    _ = manager.create_resource(data={'hourly_cost': 1, 'hourly_gain': 1, 'inventory_nr': 123},
                                resource_type=AvailableResources.MACHINE)

    # both non-optional fields not provided
    _ = manager.create_resource(data={'name': 'John', 'surname': 'Smith'},
                                resource_type=AvailableResources.EMPLOYEE)


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


if __name__ == "__main__":
    scope_usage_example()
    resources_creation_example()
    validation_errors_example()

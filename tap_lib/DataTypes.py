"""This file contains all data types used in project"""


import numpy as np
import pydantic as pdt
import typing as typ
import collections as coll
import enum


class Resource(pdt.BaseModel):
    """Informal interface for any type of resource_type that may be used in a factory"""

    hourly_cost: typ.Union[int, float]
    hourly_gain: typ.Union[int, float]

    id: pdt.conint(ge=0)


class Employee(Resource):
    """Class representing an employee in factory"""

    name: str
    surname: str

    # TODO add the rest of necessary fields


class Machine(Resource):
    """Class representing a machine in factory"""

    inventory_nr: str

    # TODO add the rest of necessary fields


class AvailableResources(enum.Enum):
    """Enum type, lists all available types of resources"""
    EMPLOYEE = Employee
    MACHINE = Machine


class ResourceManager:
    """Class that manages all types of resources within the project"""

    def __init__(self):
        self.used_ids: typ.Dict[AvailableResources: int] = coll.defaultdict(None)

    def create_resource(self, data, resource_type: AvailableResources) -> Resource:
        data['id']: int = self.update_resource_ids(resource_type=resource_type)

        # TODO add try statement, if resource is not created, delete last inserted id
        resource: Resource = self.map_resource_class(data=data, resource_type=resource_type)

        return resource

    def update_resource_ids(self, resource_type: AvailableResources) -> int:
        if resource_type not in self.used_ids:  # assign first id of provided Resource type
            new_id = 0
            self.used_ids[resource_type] = [new_id]

            return new_id

        else:  # increment id of provided type
            new_id = np.max(self.used_ids[resource_type]) + 1
            self.used_ids[resource_type].append(new_id)

            return new_id

    @classmethod
    def map_resource_class(cls, data, resource_type: AvailableResources) -> Resource:
        if resource_type == AvailableResources.MACHINE:
            return Machine(**data)

        if resource_type == AvailableResources.EMPLOYEE:
            return Employee(**data)


def resources_creation_example():
    manager = ResourceManager()  # call the manager, should be used as singleton

    employees = [manager.create_resource(data={'hourly_cost': 1, 'hourly_gain': 1,    # pass data
                                               'name': 'John', 'surname': 'Smith'},   # pass data
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
    machine = manager.create_resource(data={'hourly_cost': 1, 'hourly_gain': 1, 'inventory_nr': 123},
                                      resource_type=AvailableResources.MACHINE)

    # both non-optional fields not provided
    employee = manager.create_resource(data={'name': 'John', 'surname': 'Smith'},
                                       resource_type=AvailableResources.EMPLOYEE)


if __name__ == "__main__":
    resources_creation_example()
    validation_errors_example()

"""This file contains all data types used in project"""

from pydantic import BaseModel, conint
from typing import Union, Dict, List, Any
from collections import defaultdict
from enum import Enum

import numpy as np

NonNegativeInt = Union[int, np.int32, np.int64]
RealNumber = Union[float, np.float32, np.float64, int, np.int32, np.int64]
PositiveNumber = Union[float, np.float32, np.float64, int, np.int32, np.int64]


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
    """Class that manages all types of resources within the project. Every Resource instance should be created using
    this class."""

    def __init__(self):
        self.used_ids: Dict[AvailableResources, int] = defaultdict(None)

    def create_resource(self, data, resource_type: AvailableResources) -> Resource:
        # TODO make final decision on id problem, taken from database or assigned dynamically or both(third option
        #  makes least sense for project of this scale)
        data['id']: int = self.__update_resource_ids(resource_type=resource_type)

        # TODO add try statement, if resource is not created, delete last inserted id
        resource: Resource = self.map_resource_class(data=data, resource_type=resource_type)

        return resource

    def __update_resource_ids(self, resource_type: AvailableResources) -> int:
        if resource_type not in self.used_ids:  # assign first id of provided Resource type
            new_id = 0
            self.used_ids[resource_type]: List[int] = [new_id]

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


class FactoryAssignmentScheduleError(Exception):
    def __init__(self, msg: str = None, value: Any = None):
        super().__init__(msg)
        self.msg: str = msg
        self.value: Any = value


class FactoryAssignmentSchedule(np.ndarray):
    """
    Class representing solution of factory assignment problem.
    Documentation: https://numpy.org/doc/stable/user/basics.subclassing.html#
    """

    def __new__(cls, input_array=None, machines: List[Machine] = None, employees: List[Employee] = None,
                time_span: List[PositiveNumber] = None, encountered_it: NonNegativeInt = None,
                allowed_values: List[Any] = None, dtype=None) -> object:
        """
        Function creates new instance of class, assigns extra properties and returns created obj.
        @param input_array: input data, any form convertable to an array
        @type input_array: array_like
        @param machines: machines in schedule
        @type machines: List[Machine]
        @param employees: employees in schedule
        @type employees: List[Employee]
        @param time_span: time period as vector in schedule
        @type time_span: List[PositiveNumber]
        @param encountered_it: iteration of main loop of algorithm at which this solution was encountered
        @type encountered_it: NonNegativeInt
        @param allowed_values: list of values allowed within the matrix
        @type allowed_values: List[Any]
        @param dtype: data type of elements of schedule assignment
        @type dtype: data-type
        @return obj: new instance of FactoryAssignmentSchedule
        @rtype obj: object
        """

        obj = cls.__factory__(input_array=input_array, machines=machines, employees=employees, time_span=time_span,
                              dtype=dtype)
        obj.__machines = machines
        obj.__employees = employees
        obj.__time_span = time_span

        if encountered_it is not None:
            obj.__encountered_it = encountered_it

        if allowed_values is not None:
            obj.__allowed_values = allowed_values

        return obj

    def __setitem__(self, key, value):
        """
        Function prevents assignment out of range provided in __new__.
        @param key:
        @type key: Iterable
        @param value: value to be assigned to matrix
        @type value: Any
        """
        if value not in self.allowed_values:
            raise FactoryAssignmentScheduleError(msg='tried to assign value not allowed by allowed_values', value=value)

        super().__setitem__(key, value)

    @classmethod
    def __factory__(cls, input_array, machines: List[Machine], employees: List[Employee],
                    time_span: List[PositiveNumber], dtype) -> object:
        """
        Function validates the input parameters and returns numpy array_like obj.
        If input_array is provided, validate the dimensions along according axis and return obj(functionality not
        implemented yet).
        If input_array is not provided, create 3-dimensional matrix of ones(every assignment possible) and return obj.
        @param input_array: input data, any form convertable to an array
        @type input_array: array_like
        @param machines: machines in schedule
        @type machines: List[Machine]
        @param employees: employees in schedule
        @type employees: List[Employee]
        @param time_span: time period as vector in schedule
        @type time_span: List[PositiveNumber]
        @param dtype: data type of elements of schedule assignment
        @type dtype: data-type
        @return: 3-dimensional array_like obj
        @rtype: object
        """

        if input_array is None:  # if input_array is not provided, create 3-dimensional matrix of ones and return
            if machines is None:  # first dimension not provided
                raise FactoryAssignmentScheduleError(msg='Both input_array and machines are None', value=machines)

            if employees is None:  # second dimension not provided
                raise FactoryAssignmentScheduleError(msg='Both input_array and employees are None', value=employees)

            if time_span is None:  # third dimension not provided
                raise FactoryAssignmentScheduleError(msg='Both input_array and time_span are None', value=time_span)

            # obj has to be view type
            obj = np.ones((len(machines), len(employees), len(time_span)), dtype=dtype).view(cls)
            return obj

        # TODO decide if this functionality should be implemented or delete code below
        # code below should be used and further implemented if creation by input_array be needed
        # else:  # create
        #     obj = np.asarray(input_array, dtype=dtype).view(cls)  # obj has to be view type
        #
        #     if np.any((obj != 0) & (obj != 1)):
        #         raise FactoryAssignmentScheduleError(msg='input_array must contain only 0 or 1', value=input_array)
        #
        #     if len(obj.shape) != 3:
        #         raise FactoryAssignmentScheduleError(msg='input_array must be 3-dimensional', value=input_array)
        #
        #     return obj

        raise FactoryAssignmentScheduleError(msg='FactoryAssignmentSchedule.__instance_factory() end reached without '
                                                 'returning a new instance')

    def __array_finalize__(self, obj) -> None:
        # noinspection GrazieInspection
        """
        Function performed by numpy on array after assignment has been finished.
        @param obj: right side of assignment operator
        @type obj: object
        @return: None
        @rtype: None
        """
        if obj is None:
            return

        # TODO implement loss of dimension information when slice is created(mach/empl/time dim lost when slice called)
        self.__machines: List[Machine] = getattr(obj, 'machines', None)
        self.__employees: List[Employee] = getattr(obj, 'employees', None)
        self.__time_span: List[PositiveNumber] = getattr(obj, 'time_span', None)
        self.__encountered_it: NonNegativeInt = getattr(obj, 'encountered_it', None)
        self.__allowed_values: List[Any] = getattr(obj, 'allowed_values', None)

        self.__cost: PositiveNumber = self.__evaluate_cost__()

    def __evaluate_cost__(self) -> PositiveNumber:
        """
        Function evaluates cost of schedule.
        Called only after all the changed elements within the matrix had been assigned.
        """

        # TODO implement proper evaluation
        return np.sum(self.view(np.ndarray))  # self.view(np.ndarray) necessary to avoid infinite recursion

    @property
    def machines(self) -> List[Machine]:
        return self.__machines

    @machines.setter
    def machines(self, value: List[Machine]) -> None:
        raise FactoryAssignmentScheduleError(msg='machines is a read-only property')

    @property
    def employees(self) -> List[Employee]:
        return self.__employees

    @employees.setter
    def employees(self, value: List[Employee]) -> None:
        raise FactoryAssignmentScheduleError(msg='employees is a read-only property')

    @property
    def time_span(self) -> List[PositiveNumber]:
        return self.__time_span

    @time_span.setter
    def time_span(self, value: List[PositiveNumber]) -> None:
        raise FactoryAssignmentScheduleError(msg='time_span is a read-only property')

    @property
    def encountered_it(self) -> NonNegativeInt:
        return self.__encountered_it

    @encountered_it.setter
    def encountered_it(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError('encountered_it must be of type int')

        if value < 0:
            raise ValueError('encountered_it must be non-negative')

        # noinspection PyAttributeOutsideInit
        self.__encountered_it: NonNegativeInt = value

    @property
    def cost(self) -> PositiveNumber:
        return self.__cost

    @cost.setter
    def cost(self, value) -> None:
        raise FactoryAssignmentScheduleError('cost is read-only parameter')

    @property
    def allowed_values(self) -> List[Any]:
        return self.__allowed_values

    @allowed_values.setter
    def allowed_values(self, value: List[Any]):
        raise FactoryAssignmentScheduleError(msg='allowed_values is a read-only property', value=value)


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


def solution_usage_example():
    res_manager = ResourceManager()
    machines = [res_manager.create_resource(data={'hourly_cost': 1, 'hourly_gain': 1, 'inventory_nr': 11},
                                            resource_type=AvailableResources.MACHINE)]
    employees = [res_manager.create_resource(data={'hourly_cost': 1, 'hourly_gain': 1, 'name': 'John',
                                                   'surname': 'Smith'}, resource_type=AvailableResources.EMPLOYEE),
                 res_manager.create_resource(data={'hourly_cost': 2, 'hourly_gain': 2, 'name': 'Andrew',
                                                   'surname': 'Allen'}, resource_type=AvailableResources.EMPLOYEE)]
    time_span = [0, 1]
    sol: FactoryAssignmentSchedule = FactoryAssignmentSchedule(machines=machines, employees=employees,
                                                               time_span=time_span, encountered_it=2,
                                                               allowed_values=[0, 1], dtype='int32')

    sol_view: FactoryAssignmentSchedule = sol[:, :, 0]  # take assignments of first time period

    print(sol)
    print(sol_view)
    print()
    print(sol.shape)
    print(sol_view.shape)
    print()
    print(sol.machines)
    print(sol_view.machines)
    print()
    print(sol.employees)
    print(sol_view.employees)
    print()
    print(sol.time_span)
    print(sol_view.time_span)
    print()
    print(sol.encountered_it)
    print(sol_view.encountered_it)
    print()
    print(sol.allowed_values)
    print(sol_view.allowed_values)
    print()
    print(sol.cost)
    print(sol_view.cost)
    print()
    print(sol.dtype)
    print(sol_view.dtype)


if __name__ == "__main__":
    # solution_usage_example()
    # resources_creation_example()
    # validation_errors_example()
    pass

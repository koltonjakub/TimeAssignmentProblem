"""This file contains all data types used in project"""

from pydantic import BaseModel, conint, confloat, validator, ValidationError
from typing import Union, Dict, List, Any, Tuple, Iterable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from json import load

import numpy as np


class Resource(BaseModel):
    """Informal interface for any type of resource_type that may be used in a factory"""
    id: conint(ge=0, strict=True)  # corresponds with dimension in schedule matrix


class Machine(Resource):
    """Class representing a machine in factory"""
    hourly_cost: confloat(ge=0)
    hourly_gain: confloat(ge=0)
    inventory_nr: conint(ge=0, strict=True)


# noinspection PyMethodParameters
class Employee(Resource):
    """Class representing an employee in factory"""
    hourly_cost: confloat(ge=0)
    hourly_gain: Dict[conint(ge=0, strict=True), confloat(ge=0)]
    name: str
    surname: str

    @validator('hourly_gain', pre=True)
    def convert_keys_to_int_if_possible(cls, v):
        """
        Function validates and converts keys of hourly_gain dictionary if conversion is possible. Conversion is
        necessary due to the format in which .json stores int keys which is string.
        @param v: hourly_gain structure that maps employee ti theirs extra production on certain machine
        @type v: Dict[str: float]
        @return: hourly_gain with keys converted to int
        @rtype: Dict[int: int]
        """
        return {int(key) if isinstance(key, str) and key.isdigit() else key: value for key, value in v.items()}


class TimeSpan(Resource):
    """Class representing one hour of simulation"""
    datetime: datetime

    # noinspection PyMethodParameters
    @validator('datetime')
    def validate_some_datetime(cls, value):
        """
        Function validates that the provided datetime are within the working hours of the factory and every unit is
        strictly one hour.
        @param value: datetime of simulation unit
        @type value: datetime
        @return: valid datetime
        @rtype: datetime
        """
        if not (6 <= value.hour <= 23) or value.minute != 0 or value.second != 0 or value.microsecond != 0:
            raise ValidationError(f'invalid time: h={value.hour}, m={value.minute}, s={value.second}, '
                                  f'ms={value.microsecond}')
        return value


class AvailableResources(Enum):
    """Enum type, lists of all available types of resources"""
    EMPLOYEE = Employee
    MACHINE = Machine
    TIME = TimeSpan


@dataclass
class ResourceContainer:
    """Associative container for all types of Resource subclasses"""
    machines: List[Machine] = field(default_factory=list)
    employees: List[Employee] = field(default_factory=list)
    time_span: List[TimeSpan] = field(default_factory=list)

    def __post_init__(self):
        for field_name, field_value in self.__dict__.items():
            type(self).validate_field(field_name, field_value)

    def __setattr__(self, name, value):
        type(self).validate_field(name, value)
        super().__setattr__(name, value)

    @classmethod
    def validate_field(cls, field_name, field_value) -> None:
        cls.validate_machines(field_name, field_value)
        cls.validate_employees(field_name, field_value)
        cls.validate_time_span(field_name, field_value)

    @staticmethod
    def validate_machines(field_name: str, field_value: List[Machine]) -> None:
        if field_name != "machines":
            return

        if not isinstance(field_value, List):
            raise TypeError('machines must be a List')

        if not all(isinstance(value, Machine) for value in field_value):
            raise ValueError(f"All values in {field_name} must be a Machine class")

    @staticmethod
    def validate_employees(field_name: str, field_value: List[Machine]) -> None:
        if field_name != "employees":
            return

        if not isinstance(field_value, List):
            raise TypeError('employees must be a List')

        if not all(isinstance(value, Employee) for value in field_value):
            raise ValueError(f"All values in {field_name} must be a Employee class")

    @staticmethod
    def validate_time_span(field_name: str, field_value: List[TimeSpan]) -> None:
        if field_name != "time_span":
            return

        if not isinstance(field_value, List):
            raise TypeError('time_span must be a List')

        if not all(isinstance(value, TimeSpan) for value in field_value):
            raise ValueError(f"All values in {field_name} must be a TimeSpan class")


class ResourceImportError(Exception):
    def __init__(self, msg: str = None, value: Any = None) -> None:
        super().__init__(msg)
        self.msg: str = msg
        self.value: Any = value


class ResourceManager:
    """This class provides a user with utilities for importing and validating project databases."""

    @staticmethod
    def import_resources_from_json(file_path: str) -> ResourceContainer | None:
        """Method imports resources from .json file and returns them as ResourceContainer.
        @param file_path: path to .json file
        @type file_path: str
        """
        try:
            with open(file_path, "r") as file:
                data = load(file)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None

        try:
            machines = [Machine(**machine) for machine in data.get("machines", [])]
            employees = [Employee(**employee) for employee in data.get("employees", [])]
            time_span = [TimeSpan(**ts) for ts in data.get("time_span", [])]

            resources: ResourceContainer = ResourceContainer(machines=machines, employees=employees,
                                                             time_span=time_span)
        except ValidationError as e:
            print(f'Error: {e}')
            return None

        try:
            ResourceManager.validate_ids(resources)
        except ResourceImportError as e:
            print(f"Error: {e}")
            return None

        return resources

    @staticmethod
    def validate_ids(imp_resources: ResourceContainer) -> None:
        """Method validates that all ids meet the required standard(ints from 0 to N, distinct for each resource type)
        @param imp_resources: imported resources
        @type imp_resources: ResourceContainer
        """
        if list(set([mach.id for mach in imp_resources.machines])) != [mach.id for mach in imp_resources.machines]:
            raise ResourceImportError(msg='machines ids are not distinct')

        if list(set([empl.id for empl in imp_resources.employees])) != [empl.id for empl in imp_resources.employees]:
            raise ResourceImportError(msg='employees ids are not distinct')

        if list(set([time.id for time in imp_resources.time_span])) != [time.id for time in imp_resources.time_span]:
            raise ResourceImportError(msg='time_span ids are not distinct')


class FactoryAssignmentScheduleError(Exception):
    def __init__(self, msg: str = None, value: Any = None) -> None:
        super().__init__(msg)
        self.msg: str = msg
        self.value: Any = value


class FactoryAssignmentSchedule(np.ndarray):
    """
    Class representing solution of factory assignment problem.
    Documentation: https://numpy.org/doc/stable/user/basics.subclassing.html#
    """

    def __new__(cls, machines: List[Machine], employees: List[Employee], time_span: List[TimeSpan],
                input_array: object = None, allowed_values: List[Any] = None, encountered_it: int = None,
                dtype: object = None) -> 'FactoryAssignmentSchedule':
        """
        Function creates new instance of class, assigns extra properties and returns created obj.
        @param input_array: input data, any form convertable to an array
        @type input_array: array_like
        @param machines: machines in schedule
        @type machines: List[Machine]
        @param employees: employees in schedule
        @type employees: List[Employee]
        @param time_span: time period as vector in schedule
        @type time_span: List[TimeSpan]
        @param encountered_it: iteration of main loop of algorithm at which this solution was encountered
        @type encountered_it: int
        @param allowed_values: list of values allowed within the matrix
        @type allowed_values: List[Any]
        @param dtype: data type of elements of schedule assignment
        @type dtype: data-type
        @return obj: newly created instance
        @rtype obj: FactoryAssignmentSchedule
        """

        obj = cls.__factory__(input_array=input_array, machines=machines, employees=employees, time_span=time_span,
                              dtype=dtype)
        obj.__machines = machines
        obj.__employees = employees
        obj.__time_span = time_span
        obj.allowed_values = allowed_values

        if encountered_it is not None:
            obj.__encountered_it = encountered_it

        return obj

    def __getitem__(self, item) -> Any:
        """
        Function handles standard __getitem__ utilities, performs reshape is a slice of FactoryAssignmentSchedule is
        taken and then slices the corresponding ResourceList attributes.
        @param item:
        @type item:
        @return:
        @rtype:
        """
        obj = super().__getitem__(item)

        if np.isscalar(obj):
            return obj

        if obj.ndim == 0:
            return obj.item()

        machine_sl, employees_sl, time_span_sl = item

        def calculate_dim(dimension_list: List[Union[Machine, Employee, TimeSpan]], dimension_slice: slice) -> int:
            """
            Function returns the length of sliced dimension_list as int based on the params.
            @param dimension_list: List to be sliced
            @type dimension_list: List[Union[Machine, Employee, TimeSpan]]
            @param dimension_slice: slice of dimension_list
            @type dimension_slice: slice
            @return: calculated dimension
            @rtype: int
            """
            if isinstance(dimension_slice, int):
                return 1
            elif isinstance(dimension_slice, slice):
                if dimension_slice.start is None and dimension_slice.stop is None and dimension_slice.step is None:
                    return len(dimension_list)
                elif (dimension_slice.start is None and dimension_slice.stop is not None and
                      dimension_slice.step is None):
                    return dimension_slice.stop
                elif (dimension_slice.start is not None and dimension_slice.stop is not None and
                      dimension_slice.step is None):
                    return dimension_slice.stop - dimension_slice.start
                elif (dimension_slice.start is not None and dimension_slice.stop is not None and
                      dimension_slice.step is not None):
                    return (dimension_slice.stop - dimension_slice.start) // 2
                else:
                    raise FactoryAssignmentScheduleError(msg='dim could not be calculated', value=dimension_slice)

        preserved_shape = (
            calculate_dim(dimension_list=self.machines, dimension_slice=machine_sl),
            calculate_dim(dimension_list=self.employees, dimension_slice=employees_sl),
            calculate_dim(dimension_list=self.time_span, dimension_slice=time_span_sl)
        )

        obj = obj.reshape(preserved_shape)
        obj.__machines = self.machines[machine_sl]
        obj.__employees = self.employees[employees_sl]
        obj.__time_span = self.time_span[time_span_sl]

        if obj.ndim != 3:  # check if 3d shape is preserved
            raise FactoryAssignmentScheduleError(msg=f'obj.ndim invalid: {obj.ndim} != 3', value=obj)

        return obj

    def __setitem__(self, key, value) -> None:
        """
        Function prevents assignment out of range provided in __new__ on top of standard utility of __setitem__.
        @param key:
        @type key: Iterable
        @param value: value to be assigned to matrix
        @type value: Any
        """

        # if isinstance(value, Iterable) and np.any([elem not in self.allowed_values for elem in value]):
        #     raise FactoryAssignmentScheduleError(msg='tried to assign not allowed value', value=value)

        if np.isscalar(value):
            if value not in self.allowed_values:
                raise FactoryAssignmentScheduleError(msg=f'{value} is not in allowed_values', value=value)

        super().__setitem__(key, value)

    @classmethod
    def __factory__(cls, input_array, machines: List[Machine], employees: List[Employee],
                    time_span: List[TimeSpan], dtype) -> 'FactoryAssignmentSchedule':
        """
        Function validates the input parameters and returns numpy array_like obj.
        If input_array is provided, validate the dimensions along according axis and return obj(functionality not
        implemented yet).
        If input_array is not provided, create 3-dimensional matrix of ones(every assignment possible) and return obj.
        @param input_array: input data, any form convertable to a numpy array
        @type input_array: array_like
        @param machines: machines in schedule
        @type machines: List[Machine]
        @param employees: employees in schedule
        @type employees: List[Employee]
        @param time_span: vector of simulation time in schedule
        @type time_span: List[TimeSpan]
        @param dtype: data type of elements of schedule assignment
        @type dtype: data-type
        @return: 3-dimensional FactoryAssignmentSchedule instance
        @rtype: FactoryAssignmentSchedule
        """

        # read-only params need to be validated explicitly here, because of the read-only functionality, that forces
        # assignment to self.__attribute_name
        params: List[List[Union[Machine, Employee]]] = [machines, employees, time_span]
        param_types: List[object] = [Machine, Employee, TimeSpan]
        labels: List[str] = ['machines', 'employees', 'time_span']
        for param, param_type, label in zip(params, param_types, labels):
            if np.any([not isinstance(elem, param_type) for elem in param]):
                raise FactoryAssignmentScheduleError(msg=f'{label}: {param} is not {param_type}', value=param)

        if input_array is None:  # Case: explicit constructor
            enforced_shape: Tuple[int, int, int] = (len(machines), len(employees), len(time_span))
            obj = np.ones(enforced_shape, dtype=dtype).view(cls)
            return obj

        try:  # Case: array_like input
            obj = np.asarray(input_array, dtype=dtype).view(cls)
        except (TypeError, ValueError):
            raise FactoryAssignmentScheduleError(msg=f'input_array of type {type(input_array)} could not be converted '
                                                     f'to FactoryAssignmentSchedule', value=input_array)
        else:
            enforced_shape: Tuple[int, int, int] = (len(machines), len(employees), len(time_span))

            if obj.shape != enforced_shape:
                raise FactoryAssignmentScheduleError(msg=f'shape invalid: input_array.shape={obj.shape} != '
                                                         f'{enforced_shape}=enforced_shape', value=obj)
            return obj

    def __array_finalize__(self, obj) -> None:
        # noinspection GrazieInspection
        """
        Function performed by numpy on array after all assignments had been finished.
        @param obj: right side of assignment operator
        @type obj: object
        @return: None
        @rtype: None
        """
        if obj is None:  # Case: Explicit constructor called
            return

        # Case: view() or slice called
        self.__machines: List[Machine] = getattr(obj, 'machines', None)
        self.__employees: List[Employee] = getattr(obj, 'employees', None)
        self.__time_span: List[TimeSpan] = getattr(obj, 'time_span', None)
        self.__encountered_it: int = getattr(obj, 'encountered_it', None)
        self.__allowed_values: List[Any] = getattr(obj, 'allowed_values', None)
        self.__cost: float = self.__evaluate_cost__()

    # TODO implement __array_ufunc__ method if numpy will have to handle FactoryAssignmentSchedule type arrays

    def __evaluate_cost__(self) -> float:
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
    def time_span(self) -> List[TimeSpan]:
        return self.__time_span

    @time_span.setter
    def time_span(self, value: List[float]) -> None:
        raise FactoryAssignmentScheduleError(msg='time_span is a read-only property')

    @property
    def encountered_it(self) -> int:
        return self.__encountered_it

    @encountered_it.setter
    def encountered_it(self, value: int) -> None:
        if not isinstance(value, int):
            raise FactoryAssignmentScheduleError('encountered_it must be of type int')

        if value < 0:
            raise FactoryAssignmentScheduleError('encountered_it must be non-negative')

        # noinspection PyAttributeOutsideInit
        self.__encountered_it = value

    @property
    def cost(self) -> float:
        return self.__cost

    @cost.setter
    def cost(self, value) -> None:
        raise FactoryAssignmentScheduleError('cost is read-only parameter')

    @property
    def allowed_values(self) -> Iterable[Any]:
        return self.__allowed_values

    @allowed_values.setter
    def allowed_values(self, value: Iterable[Any]) -> None:
        if not isinstance(value, Iterable) or isinstance(value, str):
            raise FactoryAssignmentScheduleError(msg='allowed_values must be an iterable', value=value)

        # noinspection PyAttributeOutsideInit
        self.__allowed_values: Iterable[Any] = value

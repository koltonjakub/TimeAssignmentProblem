"""This file contains all data types and functions specified for factory"""


from typing import Union, Dict, List, Any, Tuple, Iterable
from configparser import ConfigParser
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from copy import deepcopy
from json import load

import numpy as np

import random
import os


current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
config_directory = os.path.join(parent_directory, "tap_lib", "config.ini")

config = ConfigParser()
config.read(config_directory)

WORK_DAY_DURATION = config.getint('Globals', 'WORK_DAY_DURATION_IN_HOURS')
WORK_DAY_START_HOUR = config.getint('Globals', 'WORK_DAY_START_AS_HOUR')
WORK_DAY_END_HOUR = config.getint('Globals', 'WORK_DAY_END_AS_HOUR')
STANDARD_DAY_COEFFICIENT = config.getint('Globals', 'STANDARD_DAY_COEFFICIENT')
EXCEEDED_DAY_COEFFICIENT = config.getint('Globals', 'EXCEEDED_DAY_COEFFICIENT')
MAX_TIME_SPAN_EXTENSION_OCCURRENCE = config.getint('Globals', 'MAX_TIME_SPAN_EXTENSION_OCCURRENCE')
NEIGHBOURHOOD_DIAMETER = config.getint('Globals', 'NEIGHBOURHOOD_DIAMETER')


class ResourceImportError(Exception):
    """Exception raised when a resource cannot be imported properly."""
    def __init__(self, msg: str = None, value: Any = None) -> None:
        super().__init__(msg)
        self.msg: str = msg
        self.value: Any = value


class FactoryAssignmentScheduleError(Exception):
    def __init__(self, msg: str = None, value: Any = None) -> None:
        super().__init__(msg)
        self.msg: str = msg
        self.value: Any = value


class ShiftAssignmentError(Exception):
    """Exception raised when a shift cannot be assigned to employee."""
    def __init__(self, msg: str = None, value: Any = None) -> None:
        super().__init__(msg)
        self.msg: str = msg
        self.value: Any = value


class ShiftUnassignmentError(Exception):
    """Exception raised when a shift cannot be unassigned from employee."""
    def __init__(self, msg: str = None, value: Any = None) -> None:
        super().__init__(msg)
        self.msg: str = msg
        self.value: Any = value


class InvalidTotalProductionError(Exception):
    """Exception raised when total production is invalid ."""
    def __init__(self, msg: str = None, value: Any = None) -> None:
        super().__init__(msg)
        self.msg: str = msg
        self.value: Any = value


class InvalidScheduleAssignmentError(Exception):
    """Exception raised when schedule assignment is invalid."""
    def __init__(self, msg: str = None, value: Any = None) -> None:
        super().__init__(msg)
        self.msg: str = msg
        self.value: Any = value


class GenerateStartingSolutionError(Exception):
    """Exception raised when the function cannot generate a valid starting solution."""
    def __init__(self, msg: str = None, value: Any = None) -> None:
        super().__init__(msg)
        self.msg: str = msg
        self.value: Any = value


@dataclass(frozen=True)
class Machine:
    """Class representing a machine in factory"""
    id: int
    hourly_cost: float
    hourly_gain: float
    max_workers: int
    inventory_nr: Union[str, int]
    demand: Union[float, int]

    def __post_init__(self):
        for field_name, field_value in self.__dict__.items():
            cls = type(self)
            cls.validate_id(field_name, field_value)
            cls.validate_hourly_cost(field_name, field_value)
            cls.validate_hourly_gain(field_name, field_value)
            cls.validate_max_workers(field_name, field_value)
            cls.validate_inventory_nr(field_name, field_value)
            cls.validate_demand(field_name, field_value)

    @staticmethod
    def validate_id(field_name, field_value) -> None:
        if field_name != "id":
            return
        if not isinstance(field_value, int):
            raise TypeError(f"Field must be of type int")
        if field_value < 0:
            raise ValueError(f"Field must be greater or equal to 0")

    @staticmethod
    def validate_hourly_cost(field_name, field_value) -> None:
        if field_name != "hourly_cost":
            return
        if not isinstance(field_value, (float, int)):
            raise TypeError(f"Field must be of type int or float")
        if field_value < 0:
            raise ValueError(f"Field must be greater or equal to 0")

    @staticmethod
    def validate_hourly_gain(field_name, field_value) -> None:
        if field_name != "hourly_gain":
            return
        if not isinstance(field_value, (float, int)):
            raise TypeError(f"Field must be of type int or float")
        if field_value < 0:
            raise ValueError(f"Field must be greater or equal to 0")

    @staticmethod
    def validate_max_workers(field_name, field_value) -> None:
        if field_name != "max_workers":
            return
        if not isinstance(field_value, int):
            raise TypeError(f"Field must be of type int")
        if field_value < 1:
            raise ValueError(f"Field must be greater or equal to 1")

    @staticmethod
    def validate_inventory_nr(field_name, field_value) -> None:
        if field_name != "inventory_nr":
            return
        if not isinstance(field_value, (str, int)):
            raise TypeError(f"Field must be of type str or int")

    @staticmethod
    def validate_demand(field_name, field_value) -> None:
        if field_name != "demand":
            return
        if not isinstance(field_value, (float, int)):
            raise TypeError(f"Field must be of type float or int")
        if field_value < 0:
            raise ValueError(f'Field must be greater or equal to 0')


@dataclass(frozen=True)
class Employee:
    """Class representing an employee in factory"""
    id: int
    hourly_cost: float
    hourly_gain: Dict[int, float]
    name: str
    surname: str
    shift_duration: int

    def __post_init__(self):
        for field_name, field_value in self.__dict__.items():
            cls = type(self)
            cls.validate_id(field_name, field_value)
            cls.validate_hourly_cost(field_name, field_value)
            cls.validate_hourly_gain(field_name, field_value)
            cls.validate_name(field_name, field_value)
            cls.validate_surname(field_name, field_value)
            cls.validate_shift_duration(field_name, field_value)

    @staticmethod
    def validate_id(field_name, field_value) -> None:
        if field_name != "id":
            return
        if not isinstance(field_value, int):
            raise TypeError(f"Field must be of type int")
        if field_value < 0:
            raise ValueError(f"Field must be greater or equal to 0")

    @staticmethod
    def validate_hourly_cost(field_name, field_value) -> None:
        if field_name != "hourly_cost":
            return
        if not isinstance(field_value, (float, int)):
            raise TypeError(f"Field must be of type int or float")
        if field_value < 0:
            raise ValueError(f"Field must be greater or equal to 0")

    @staticmethod
    def validate_hourly_gain(field_name, field_value) -> None:
        """Function validates key: value pair for hourly gain """
        if field_name != "hourly_gain":
            return
        if not isinstance(field_value, Dict):
            raise TypeError(f"Field must be of type Dict")

        for key, value in field_value.items():
            if not isinstance(key, int):
                raise TypeError(f"hourly_gain.key {key} must be of type int")

            if not isinstance(value, (float, int)):
                raise TypeError(f"hourly_gain.value {value} must be of type int or float")

            if int(key) < 0:
                raise ValueError(f"hourly_gain.key {key} must be greater or equal to 0")

            if value < 0:
                raise ValueError(f"hourly_gain.value {value} must be greater or equal to 0")

    @staticmethod
    def validate_name(field_name, field_value) -> None:
        if field_name != "name":
            return
        if not isinstance(field_value, str):
            raise TypeError(f"Field must be of type str")

    @staticmethod
    def validate_surname(field_name, field_value) -> None:
        if field_name != "surname":
            return
        if not isinstance(field_value, str):
            raise TypeError(f"Field must be of type str")

    @staticmethod
    def validate_shift_duration(field_name, field_value) -> None:
        if field_name != "shift_duration":
            return
        if not isinstance(field_value, int):
            raise TypeError(f"Field must be of type int")
        if field_value < 0:
            raise ValueError(f"Field must be non-negative int")


@dataclass(frozen=True)
class TimeSpan:
    """Class representing an hour of simulation."""
    id: int
    datetime: datetime

    def __post_init__(self):
        for field_name, field_value in self.__dict__.items():
            cls = type(self)
            cls.validate_id(field_name, field_value)
            cls.validate_datetime(field_name, field_value)

    @staticmethod
    def validate_id(field_name, field_value) -> None:
        if field_name != "id":
            return
        if not isinstance(field_value, int):
            raise TypeError(f"Field must be of type int")
        if field_value < 0:
            raise ValueError(f"Field must be greater or equal to 0")

    @staticmethod
    def validate_datetime(field_name, field_value) -> None:
        """Function validates that the provided datetime are within the working hours of the factory and every unit is
        strictly one hour.
        """
        if field_name != "datetime":
            return
        if not isinstance(field_value, datetime):
            raise TypeError(f"Field must be of type datetime")
        if (not (WORK_DAY_START_HOUR <= field_value.hour <= WORK_DAY_END_HOUR) or field_value.minute != 0 or
                field_value.second != 0 or field_value.microsecond != 0):
            raise ValueError(f'invalid time: h={field_value.hour}, m={field_value.minute}, '
                             f's={field_value.second}, ms={field_value.microsecond}')


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


class ResourceManager:
    """This class provides a user with utilities for importing and validating project databases."""

    @staticmethod
    def import_resources_from_json(file_path: str) -> Union[ResourceContainer, None]:
        """Method imports resources from .json file and returns them as ResourceContainer.
        @param file_path: path to .json file
        @type file_path: str
        """
        try:
            with open(file_path, "r") as file:
                data: Dict = load(file)
        except FileNotFoundError as e:
            raise e

        try:
            for empl_data in data.get("employees", []):
                empl_data["hourly_gain"] = ResourceManager.convert_keys(empl_data["hourly_gain"])
            for time_span_data in data.get("time_span", []):
                time_span_data["datetime"] = ResourceManager.convert_datetime(time_span_data["datetime"])
        except ValueError as value_error:
            raise value_error

        try:
            machines = [Machine(**machine) for machine in data.get("machines", [])]
            employees = [Employee(**employee) for employee in data.get("employees", [])]
            time_span = [TimeSpan(**ts) for ts in data.get("time_span", [])]

            resources: ResourceContainer = ResourceContainer(machines=machines, employees=employees,
                                                             time_span=time_span)
        except TypeError as type_error:
            raise type_error
        except ValueError as value_error:
            raise value_error

        try:
            ResourceManager.validate_ids(resources)
        except ResourceImportError as resource_import_error:
            raise resource_import_error

        return resources

    @staticmethod
    def convert_keys(input_dict: Dict[Union[str, int], Union[int, float]]) -> Dict[int, Union[int, float]]:
        result_dict = {}

        for key, value in input_dict.items():
            try:
                casted_key = int(key)
                if str(casted_key) == key:
                    result_dict[casted_key] = value
            except ValueError:
                raise ValueError(f'hourly_gain.key: {key} cannot be cast to int')
            else:
                result_dict[casted_key] = value
        return result_dict

    @staticmethod
    def convert_datetime(isoformat_string: str) -> datetime:
        try:
            converted_datetime: datetime = datetime.fromisoformat(isoformat_string)
        except TypeError as type_error:
            raise type_error
        else:
            return converted_datetime

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


class FactoryAssignmentSchedule(np.ndarray):
    """
    Class representing schedule of factory assignment problem.
    Documentation: https://numpy.org/doc/stable/user/basics.subclassing.html#
    """

    def __new__(cls, machines: List[Machine], employees: List[Employee], time_span: List[TimeSpan],
                input_array: object = None, allowed_values: List[Any] = None, encountered_it: int = None,
                exceeding_days: int = 0, dtype: object = None) -> 'FactoryAssignmentSchedule':
        """
        Function creates new instance of class, assigns extra properties and returns created obj.
        @param machines: machines in schedule
        @type machines: List[Machine]
        @param employees: employees in schedule
        @type employees: List[Employee]
        @param time_span: time period as vector in schedule
        @type time_span: List[TimeSpan]
        @param input_array: input data, any form convertable to an array
        @type input_array: array_like
        @param encountered_it: iteration of main loop of algorithm at which this schedule was encountered
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
        obj.__exceeding_days = exceeding_days

        if encountered_it is not None:
            obj.__encountered_it = encountered_it

        return obj

    def __getitem__(self, item) -> Union[np.ndarray[Any, Any], Any]:
        """
        Function handles standard __getitem__ utilities, performs reshape is a slice of FactoryAssignmentSchedule is
        taken and then slices the corresponding ResourceList attributes.
        @param item: slice of FactoryAssignmentSchedule object
        @type item: Tuple
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
            obj = np.zeros(enforced_shape, dtype=dtype).view(cls)
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
        self.__exceeding_days: int = getattr(obj, 'exceeding_days', None)

    def cost(self) -> float:
        """
        Function evaluates cost of schedule.
        """
        ker = np.sum(self.view(np.ndarray), axis=2)
        ker_bool = np.sum(self.view(np.ndarray), axis=1, dtype=bool)
        ker_empls = np.sum(ker, axis=0)
        ker_machs = np.sum(ker_bool, axis=1).transpose()

        costs_machs = np.multiply(ker_machs, [mach.hourly_cost for mach in self.machines])
        costs_empls = np.multiply(ker_empls, [emp.hourly_cost for emp in self.employees])

        return np.sum(costs_empls) + np.sum(costs_machs)

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
    def allowed_values(self) -> Iterable[Any]:
        return self.__allowed_values

    @allowed_values.setter
    def allowed_values(self, value: Iterable[Any]) -> None:
        if not isinstance(value, Iterable) or isinstance(value, str):
            raise FactoryAssignmentScheduleError(msg='allowed_values must be an iterable', value=value)

        # noinspection PyAttributeOutsideInit
        self.__allowed_values: Iterable[Any] = value

    @property
    def exceeding_days(self) -> int:
        return self.__exceeding_days

    @exceeding_days.setter
    def exceeding_days(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError('exceeding_days must be of type int')

        if value < 0:
            raise ValueError('exceeding_days must be non-negative')

        # noinspection PyAttributeOutsideInit
        self.__exceeding_days = value


def get_machine_maintenance(schedule: FactoryAssignmentSchedule, machine: Machine) -> Union[int, float]:
    """
    Function calculates cost incurred from working machine due to it`s maintenance spending.
    @param schedule: Schedule provided for evaluation
    @type schedule: FactoryAssignmentSchedule
    @param machine: machine producing cost
    @type machine: Machine
    @return: cost incurred by machine maintenance
    @rtype: Union[int, float]
    """
    summed_employees = np.sum(schedule[machine.id, :, :], axis=1)
    boolean_mask = summed_employees > 0
    bit_mask = boolean_mask.astype(int)
    working_hours = np.sum(bit_mask, axis=1)

    return machine.hourly_cost * working_hours


def get_employee_salary(schedule: FactoryAssignmentSchedule, employee: Employee) -> Union[int, float]:
    """
    Function calculates salary of the employee according to their`s worktime in schedule.
    @param schedule: Schedule provided for evaluation
    @type schedule: FactoryAssignmentSchedule
    @param employee: employee to be paid
    @type employee: Employee
    @return: salary of employee
    @rtype: Union[int, float]
    """
    summed_machines = np.sum(schedule[:, employee.id, :], axis=2)
    working_hours = np.sum(summed_machines, axis=0)

    return employee.hourly_cost * working_hours


def get_time_penalty(schedule: FactoryAssignmentSchedule) -> Union[int, float]:
    """
    Function calculates penalty incurred from time needed for completion of order.
    @param schedule: Schedule provided for evaluation
    @type schedule: FactoryAssignmentSchedule
    @return: evaluated penalty
    @rtype: Union[int, float]
    """
    time_span_len_in_days = len(schedule.time_span) // WORK_DAY_DURATION
    last_working_day = None

    for day in range(time_span_len_in_days - 1, -1, -1):
        if np.sum(schedule[:, :, day * WORK_DAY_DURATION: (day + 1) * WORK_DAY_DURATION]) > 0:
            last_working_day = day + 1
            break
    print(last_working_day, time_span_len_in_days)

    standard_days = last_working_day
    exceeded_days = time_span_len_in_days - last_working_day + schedule.exceeding_days
    if exceeded_days < 0:
        exceeded_days = 0  # this cancels extra penalty, if number of exceeded days is actually less than 0

    return standard_days * STANDARD_DAY_COEFFICIENT + exceeded_days * EXCEEDED_DAY_COEFFICIENT


def get_cost(schedule: FactoryAssignmentSchedule) -> Union[int, float]:
    """
    Function calculates total cost of provided schedule. The cost is sum of machines maintenance spending, employees`
    salaries and time factor.
    @param schedule: Schedule provided for evaluation of cost
    @type schedule: FactoryAssignmentSchedule
    @return: total cost
    @rtype: Union[int, float]
    """
    maintenance = np.sum([get_machine_maintenance(schedule, mach) for mach in schedule.machines])
    salary = np.sum([get_employee_salary(schedule, empl) for empl in schedule.employees])
    time_penalty = get_time_penalty(schedule)

    return maintenance + salary + time_penalty


def get_machine_production(schedule: FactoryAssignmentSchedule, machine: Machine) -> Union[int, float]:
    """
    Function calculates the machine production for a given schedule and returns it.
    Calculated as machine.hourly_gain .* schedule[machine, :, :] .* schedule.employees.hourly_gain[machine].
    @param schedule: Considered schedule
    @type schedule: FactoryAssignmentSchedule
    @param machine: Machine from the schedule
    @type machine: Machine
    @return: Total amount of production for the given machine
    @rtype: Union[int, float]
    """
    # employee_experience_matrix = np.array([empl.hourly_gain[machine.id] for empl in schedule.employees]).reshape(
    #     (len(schedule.employees), 1))
    employee_experience_matrix = np.array([empl.hourly_gain[machine.id] for empl in schedule.employees])

    schedule_of_machine = schedule[machine.id, :, :]
    hours_worked_per_employee = np.sum(schedule_of_machine, axis=2)
    production_per_employee = np.multiply(hours_worked_per_employee, employee_experience_matrix)
    machine_production = machine.hourly_gain * np.sum(production_per_employee)
    return machine_production


def get_nr_of_assigned_employees(schedule: FactoryAssignmentSchedule, machine: Machine, timespan: TimeSpan) -> int:
    """
    Function calculates number of employees assigned to the machine in given timespan.
    @param schedule: Considered schedule
    @type schedule: FactoryAssignmentSchedule
    @param machine: Machine from the schedule
    @type machine: Machine
    @param timespan: Moment in time of schedule in which the employees are assigned to the machine.
    @type timespan: TimeSpan
    @return: Number of assigned employees
    @rtype: int
    """
    return np.count_nonzero(schedule[machine.id, :, timespan.id])


def is_valid_machine_production(schedule: FactoryAssignmentSchedule, machine: Machine) -> bool:
    """
    Function checks if production demand is met for the given machine within provided schedule.
    @param schedule: Schedule to be validated
    @type schedule: FactoryAssignmentSchedule
    @param machine: Machine from the schedule to be checked
    @type machine: Machine
    @return: Is production demand met
    @rtype: bool
    """
    return get_machine_production(schedule, machine) >= machine.demand


def is_valid_total_production(schedule: FactoryAssignmentSchedule) -> bool:
    """
    Function checks if the total production in schedule meets the demand in Machines.
    @param schedule: Schedule to be validated
    @type schedule: FactoryAssignmentSchedule
    @return: Is production in schedule valid
    @rtype: bool
    """
    return np.all([is_valid_machine_production(schedule, machine) for machine in schedule.machines])


def is_valid_machine_assignment(schedule: FactoryAssignmentSchedule, machine: Machine) -> bool:
    """
    Function checks if the assignment is valid for the given machine within provided schedule.
    @param schedule: Considered schedule
    @type schedule: FactoryAssignmentSchedule
    @param machine: Machine from the schedule
    @type machine: Machine
    @return: Is assignment valid
    @rtype: bool
    """
    return np.all([get_nr_of_assigned_employees(schedule, machine, tm_sp) <=
                   machine.max_workers for tm_sp in schedule.time_span])


def is_valid_schedule_assignment(schedule: FactoryAssignmentSchedule) -> bool:
    """
    Function checks if the assignment is valid for the given schedule.
    @param schedule: Schedule to be validated
    @type schedule: FactoryAssignmentSchedule
    @return: Is employee assignment valid
    @rtype: bool
    """
    return np.all([is_valid_machine_assignment(schedule, machine) for machine in schedule.machines])


def assign_shift(schedule: FactoryAssignmentSchedule, employee: Employee, machine: Machine) -> None:
    """
    Function assigns a shift of a given employee to provided machine at first possible slot if assignment is possible.
    @param schedule: schedule to be changed
    @type schedule: FactoryAssignmentSchedule
    @param employee: employee to be assigned a shift in factory
    @type employee: Employee
    @param machine: Machine for the employee to be assigned a shift to
    @type machine: Machine
    @return: None
    @rtype:
    """
    for day in range(len(schedule.time_span) // WORK_DAY_DURATION):
        # if employee is already assigned a shift in this workday, skip
        if np.count_nonzero(schedule[:, employee.id, day * WORK_DAY_DURATION: (day + 1) * WORK_DAY_DURATION]) > 0:
            continue

        for hour in range(WORK_DAY_DURATION - employee.shift_duration):
            start_time = day * WORK_DAY_DURATION + hour
            stop_time = day * WORK_DAY_DURATION + hour + employee.shift_duration

            # check if all time slots in sequence allow for assignment of employee
            if np.all([get_nr_of_assigned_employees(schedule, machine, tm_sp) < machine.max_workers
                       for tm_sp in schedule.time_span[start_time: stop_time]]):
                schedule[machine.id, employee.id, start_time: stop_time] = np.ones(employee.shift_duration)
                return None
    raise ShiftAssignmentError(msg="Could not assign a shift due the lack of free slots in schedule",
                               value=(schedule, employee, machine))


def unassign_shift(schedule: FactoryAssignmentSchedule, employee: Employee, machine: Machine) -> None:
    """
    Function unassigns a shift of a given employee to provided machine at last possible slot if
    unassignment is possible.
    @param schedule: schedule to be changed
    @type schedule: FactoryAssignmentSchedule
    @param employee: employee to be unassigned a shift in factory
    @type employee: Employee
    @param machine: Machine for the employee to be unassigned a shift from
    @type machine: Machine
    @return: None
    @rtype:
    """
    for day in range(len(schedule.time_span) // WORK_DAY_DURATION - 1, -1, -1):
        # if employee is not assigned in this day, skip
        if (np.count_nonzero(schedule[:, employee.id, day * WORK_DAY_DURATION: (day + 1) * WORK_DAY_DURATION]) ==
                WORK_DAY_DURATION):
            continue

        for hour in range(WORK_DAY_DURATION - employee.shift_duration):
            start_time = day * WORK_DAY_DURATION + hour
            stop_time = day * WORK_DAY_DURATION + hour + employee.shift_duration

            # check if sequence counts as employee assignment
            if np.all(schedule[machine.id, employee.id, start_time: stop_time] == 1):
                schedule[machine.id, employee.id, start_time: stop_time] = np.zeros(employee.shift_duration)
                return None
    raise ShiftUnassignmentError(msg="Could not unassign a shift(not present within provided schedule)",
                                 value=(schedule, employee, machine))


def populate_machine_with_employee(schedule: FactoryAssignmentSchedule, employee: Employee, machine: Machine) -> None:
    """
    Function tries to satisfy a machine demand by continually assigning shifts for given employee.
    @param schedule: schedule to modify
    @type schedule: FactoryAssignmentSchedule
    @param employee: employee to be assigned unknown number of shifts
    @type employee: Employee
    @param machine: Machine of which the demand needs to be meet by production in schedule
    @type machine: Machine
    @return: None
    """
    while not is_valid_machine_production(schedule, machine):
        try:
            assign_shift(schedule, employee, machine)
        except ShiftAssignmentError:
            return
    return


def order_machines(machines: List[Machine]) -> List[Machine]:
    """
    Function sorts list of machines by coefficient = (hourly_gain / demand), ascending.
    @param machines: list of machines to sort
    @type machines: List[Machines]
    @return: sorted list of machines
    @rtype: List[Machines]
    """
    machines_cp = deepcopy(machines)
    machines_cp.sort(key=lambda machine: machine.hourly_gain / machine.demand)
    return machines_cp


def order_employees(employees: List[Employee], machine: Machine) -> List[Employee]:
    """
    Function sorts list of employees by coefficient hourly_gain[machine] for every employee, descending.
    @param employees:
    @type employees:
    @param machine:
    @type machine:
    @return: list of sorted employees
    @rtype: List[Employee]
    """
    employees_cp = deepcopy(employees)
    employees_cp.sort(key=lambda employee: employee.hourly_gain[machine.id], reverse=True)
    return employees_cp


def populate_schedule(schedule: FactoryAssignmentSchedule) -> None:
    """
    Function tries to satisfy every demand of the whole schedule by continually satisfying the demand of every machine.
    Appropriate errors are raised if the schedule does not meet the requirements.
    @param schedule: schedule to modify
    @type schedule: FactoryAssignmentSchedule
    @return: None
    """
    ord_machines = order_machines(schedule.machines)
    for machine in ord_machines:
        ord_employees = order_employees(schedule.employees, machine)

        for employee in ord_employees:
            populate_machine_with_employee(schedule, employee, machine)
            if is_valid_machine_production(schedule, machine):
                break

    if not is_valid_total_production(schedule):
        raise InvalidTotalProductionError(msg='Total production demand is not valid', value=schedule)

    if not is_valid_schedule_assignment(schedule):
        raise InvalidScheduleAssignmentError(msg='Schedule assignment  of employees is not valid', value=schedule)


def extend_time_span(time_span: List[TimeSpan]) -> List[TimeSpan]:
    """
    Function extends the time span by adding one day, which consists of WORK_DAY_DURATION number of elements
    @param time_span: time span to be extended
    @type time_span: List[TimeSpan]
    @return: extended time span
    @rtype: List[TimeSpan]
    """
    time_span_cp = deepcopy(time_span)
    time_span_cp.extend([TimeSpan(id=last_day_and_hour.id + WORK_DAY_DURATION,
                                  datetime=(last_day_and_hour.datetime + timedelta(days=1)))
                         for last_day_and_hour in time_span[-WORK_DAY_DURATION::]])
    return time_span_cp


def generate_starting_solution(database_path: str) -> FactoryAssignmentSchedule:
    """
    Function generates a factory assignment schedule that meets the demand stored in database. If populate_schedule
    function is not able to generate valid assignment schedule, the time_span is extended and populating of the schedule
    is performed again.

    @param database_path: path to the database file in .json format.
    @type database_path: str
    @return schedule: schedule to modify
    @rtype schedule: FactoryAssignmentSchedule
    """
    factory_resources = ResourceManager.import_resources_from_json(database_path)
    failures_counter = 0

    while failures_counter <= MAX_TIME_SPAN_EXTENSION_OCCURRENCE:
        try:
            schedule = FactoryAssignmentSchedule(
                machines=factory_resources.machines,
                employees=factory_resources.employees,
                time_span=factory_resources.time_span,
                allowed_values=[0, 1],
                encountered_it=0,
                exceeding_days=failures_counter
            )
            populate_schedule(schedule)
        except InvalidScheduleAssignmentError:
            raise
        except InvalidTotalProductionError:
            failures_counter += 1
            factory_resources.time_span = extend_time_span(factory_resources.time_span)

        else:
            return schedule
    raise GenerateStartingSolutionError(msg=f'The factory assignment schedule could not be generated within '
                                            f'{failures_counter} extensions of time_span. Check the provided database.',
                                        value=database_path)


def perform_random_sub_step(schedule: FactoryAssignmentSchedule) -> None:
    """
    Function calls assign_shift or unassign_shift on a schedule object with random employee and machine.
    @param schedule: schedule to be changed
    @type schedule: FactoryAssignmentSchedule
    @return: None
    """

    random_machine = random.choice(schedule.machines)
    random_employee = random.choice(schedule.employees)

    option_flag = random.randint(a=0, b=1)
    if option_flag == 0:
        try:
            unassign_shift(schedule, random_employee, random_machine)
        except ShiftUnassignmentError:
            return
    elif option_flag == 1:
        try:
            assign_shift(schedule, random_employee, random_machine)
        except ShiftAssignmentError:
            return
    else:
        raise InvalidScheduleAssignmentError(msg='Unsupported option_flag value.', value=option_flag)


def random_neighbour(schedule: FactoryAssignmentSchedule) -> FactoryAssignmentSchedule:
    """
    Function generates instance of FactoryAssignmentSchedule that is randomly different from provided schedule. Works
    on copy, therefore the rest of the functions perform changes in place.
    @param schedule: base schedule
    @type schedule: FactoryAssignmentSchedule
    @return: new neighbour
    @rtype: FactoryAssignmentSchedule
    """

    schedule_cp = deepcopy(schedule)

    number_of_sub_steps = random.randint(a=1, b=NEIGHBOURHOOD_DIAMETER)
    for _ in range(number_of_sub_steps):
        perform_random_sub_step(schedule_cp)

    return schedule_cp

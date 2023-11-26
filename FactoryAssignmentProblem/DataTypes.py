"""This file contains all data types used in project"""

from typing import Union, Dict, List, Any, Tuple, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from json import load

import numpy as np
from numpy import ndarray


@dataclass(frozen=True)
class Machine:
    """Class representing a machine in factory"""
    id: int
    hourly_cost: float
    hourly_gain: float
    max_workers: int
    inventory_nr: Union[str, int]

    def __post_init__(self):
        for field_name, field_value in self.__dict__.items():
            cls = type(self)
            cls.validate_id(field_name, field_value)
            cls.validate_hourly_cost(field_name, field_value)
            cls.validate_hourly_gain(field_name, field_value)
            cls.validate_max_workers(field_name, field_value)
            cls.validate_inventory_nr(field_name, field_value)

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


@dataclass(frozen=True)
class Employee:
    """Class representing an employee in factory"""
    id: int
    hourly_cost: float
    hourly_gain: Dict[int, float]
    name: str
    surname: str

    def __post_init__(self):
        for field_name, field_value in self.__dict__.items():
            cls = type(self)
            cls.validate_id(field_name, field_value)
            cls.validate_hourly_cost(field_name, field_value)
            self.validate_hourly_gain(field_name, field_value)
            cls.validate_name(field_name, field_value)
            cls.validate_surname(field_name, field_value)

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
        if (not (6 <= field_value.hour <= 23) or field_value.minute != 0 or
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


class ResourceImportError(Exception):
    def __init__(self, msg: str = None, value: Any = None) -> None:
        super().__init__(msg)
        self.msg: str = msg
        self.value: Any = value


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
        @param machines: machines in schedule
        @type machines: List[Machine]
        @param employees: employees in schedule
        @type employees: List[Employee]
        @param time_span: time period as vector in schedule
        @type time_span: List[TimeSpan]
        @param input_array: input data, any form convertable to an array
        @type input_array: array_like
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

    def __getitem__(self, item) -> Union[ndarray[Any, Any], Any]:
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

    # TODO implement __array_ufunc__ method if numpy will have to handle FactoryAssignmentSchedule type arrays

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

        return np.sum(np.add(costs_empls, costs_machs))

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

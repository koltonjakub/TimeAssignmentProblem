"""This file contains all tests designed for FactoryAssignmentProblem module from SimulatedAnnealing package."""

from unittest import TestCase, main
from typing import List, Union
from itertools import product
from datetime import datetime
from json import load

from FactoryAssignmentProblem.DataTypes import (
    Machine, Employee, TimeSpan, ResourceContainer, ResourceImportError, ResourceManager,
    FactoryAssignmentSchedule, FactoryAssignmentScheduleError,
    get_machine_production, get_nr_of_assigned_employees,
    validate_machine_production, validate_total_production,
    validate_machine_assignment, validate_schedule_assignment,
    assign_shift, unassign_shift, random_neighbour)

import numpy as np
import os

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
test_database_path = os.path.join(parent_directory, "data", "test_database.json")
invalid_machines_database_path = os.path.join(parent_directory, "data", "invalid_machine_ids_database.json")
invalid_employees_database_path = os.path.join(parent_directory, "data", "invalid_employee_ids_database.json")
invalid_time_span_database_path = os.path.join(parent_directory, "data", "invalid_time_span_ids_database.json")
test_invalid_production_database_path = os.path.join(parent_directory, "data", "test_invalid_production_database.json")
test_valid_production_database_path = os.path.join(parent_directory, "data", "test_valid_production_database.json")


# noinspection PyTypeChecker
class MachineTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(MachineTests, self).__init__(*args, **kwargs)

    def test_fields(self) -> None:
        machine = Machine(id=1, hourly_cost=0, hourly_gain=1.1, max_workers=1, inventory_nr=123, demand=1)

        self.assertEqual(machine.id, 1)
        self.assertEqual(machine.hourly_cost, 0)
        self.assertEqual(machine.hourly_gain, 1.1)
        self.assertEqual(machine.max_workers, 1)
        self.assertEqual(machine.inventory_nr, 123)
        self.assertEqual(machine.demand, 1)

    def test_id(self) -> None:
        valid_inputs = [0, 1, 2]
        for vld_inp in valid_inputs:
            self.assertEqual(Machine(id=vld_inp, hourly_cost=1.0, hourly_gain=1.0, max_workers=1, inventory_nr=123, demand=1).id,
                             vld_inp)

        invalid_inputs = [-1, -2]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValueError):
                Machine(id=inv_inp, hourly_cost=1.0, hourly_gain=1.0, max_workers=1, inventory_nr=123, demand=1)

        invalid_inputs = [-1.1, 2.0, 1.0, 1.1]
        for inv_inp in invalid_inputs:
            with self.assertRaises(TypeError):
                Machine(id=inv_inp, hourly_cost=1.0, hourly_gain=1.0, max_workers=1, inventory_nr=123, demand=1)

    def test_hourly_cost(self) -> None:
        valid_inputs = [0, 0.0, 1, 1.1]
        for vld_inp in valid_inputs:
            self.assertEqual(Machine(id=1,
                                     hourly_cost=vld_inp, hourly_gain=1.0, max_workers=1, inventory_nr=123, demand=1).hourly_cost,
                             vld_inp)

        invalid_inputs = [-7, -1]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValueError):
                Machine(id=1, hourly_cost=inv_inp, hourly_gain=1.0, max_workers=1, inventory_nr=123, demand=1)

        invalid_inputs = ['str', float]
        for inv_inp in invalid_inputs:
            with self.assertRaises(TypeError):
                Machine(id=1, hourly_cost=inv_inp, hourly_gain=1.0, max_workers=1, inventory_nr=123, demand=1)

    def test_hourly_gain(self) -> None:
        valid_inputs = [0, 0.0, 1, 1.1]
        for vld_inp in valid_inputs:
            self.assertEqual(Machine(id=1, hourly_cost=1, hourly_gain=vld_inp, max_workers=1,
                                     inventory_nr=123, demand=1).hourly_gain, vld_inp)

        invalid_inputs = [-7, -1]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValueError):
                Machine(id=1, hourly_cost=1, hourly_gain=inv_inp, max_workers=1, inventory_nr=123, demand=1)

        invalid_inputs = ['str', float]
        for inv_inp in invalid_inputs:
            with self.assertRaises(TypeError):
                Machine(id=1, hourly_cost=1.0, hourly_gain=inv_inp, max_workers=1, inventory_nr=123, demand=1)

    def test_max_workers(self) -> None:
        self.assertEqual(Machine(id=1, hourly_cost=1, hourly_gain=1.0, max_workers=1, inventory_nr=123, demand=1).max_workers, 1)

        with self.assertRaises(ValueError):
            Machine(id=1, hourly_cost=1, hourly_gain=1.0, max_workers=0, inventory_nr=123, demand=1)
        with self.assertRaises(TypeError):
            Machine(id=1, hourly_cost=1, hourly_gain=1.0, max_workers='1', inventory_nr=123, demand=1)

    def test_inventory_nr(self) -> None:
        valid_inputs = [0, 1, '123', 'machine_1']
        for vld_inp in valid_inputs:
            self.assertEqual(Machine(id=1, hourly_cost=1, hourly_gain=1.0, max_workers=1,
                                     inventory_nr=vld_inp, demand=1).inventory_nr, vld_inp)

        invalid_inputs = [lambda: 0, {1: 2}, [1, 2]]
        for inv_inp in invalid_inputs:
            with self.assertRaises(TypeError):
                Machine(id=1, hourly_cost=1.0, hourly_gain=1.0, max_workers=1, inventory_nr=inv_inp, demand=1)

    def test_demand(self) -> None:
        valid_inputs = [0, 1, 2, 2.5, 3.5]
        for vld_inp in valid_inputs:
            self.assertEqual(Machine(id=1, hourly_cost=1, hourly_gain=1.0, max_workers=1,
                                     inventory_nr=123, demand=vld_inp).demand, vld_inp)

        invalid_inputs = [lambda: 0, {1: 2}, [1, 2]]
        for inv_inp in invalid_inputs:
            with self.assertRaises(TypeError):
                Machine(id=1, hourly_cost=1.0, hourly_gain=1.0, max_workers=1, inventory_nr=123, demand=inv_inp)

        invalid_inputs = [-1, -1.1]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValueError):
                Machine(id=1, hourly_cost=1.0, hourly_gain=1.0, max_workers=1, inventory_nr=123, demand=inv_inp)


# noinspection PyTypeChecker
class EmployeeTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(EmployeeTests, self).__init__(*args, **kwargs)

    def test_fields(self) -> None:
        employee = Employee(id=1, hourly_cost=0, hourly_gain={1: 1}, name='John', surname='Ally', shift_duration=4)

        self.assertEqual(employee.id, 1)
        self.assertEqual(employee.hourly_cost, 0)
        self.assertEqual(employee.hourly_gain, {1: 1})
        self.assertEqual(employee.name, 'John')
        self.assertEqual(employee.surname, 'Ally')
        self.assertEqual(employee.shift_duration, 4)

    def test_id(self) -> None:
        valid_inputs = [0, 1, 2]
        for vld_inp in valid_inputs:
            self.assertEqual(Employee(id=vld_inp, hourly_cost=1.0, hourly_gain={1: 1.0}, name='John',
                                      surname='Ally', shift_duration=8).id, vld_inp)

        invalid_inputs = [-1.1, 1.1, 0.0]
        for inv_inp in invalid_inputs:
            with self.assertRaises(TypeError):
                Employee(id=inv_inp, hourly_cost=1.0, hourly_gain={1: 1.0}, name='John', surname='Ally',
                         shift_duration=8)

        invalid_inputs = [-1, -2]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValueError):
                Employee(id=inv_inp, hourly_cost=1.0, hourly_gain={1: 1.0}, name='John', surname='Ally',
                         shift_duration=8)

    def test_hourly_cost(self) -> None:
        valid_inputs = [0, 0.0, 1, 1.1]
        for vld_inp in valid_inputs:
            self.assertEqual(Employee(id=1, hourly_cost=vld_inp, hourly_gain={1: 1.0}, name='John',
                                      surname='Ally', shift_duration=8).hourly_cost, vld_inp)

        invalid_inputs = ['str', lambda: 'str']
        for inv_inp in invalid_inputs:
            with self.assertRaises(TypeError):
                Employee(id=1, hourly_cost=inv_inp, hourly_gain={1: 1.0}, name='John', surname='Ally', shift_duration=8)

        invalid_inputs = [-1, -1.1]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValueError):
                Employee(id=1, hourly_cost=inv_inp, hourly_gain={1: 1.0}, name='John', surname='Ally', shift_duration=8)

    def test_hourly_gain(self) -> None:
        valid_inputs = [{0: 1}, {0: 1.1}, {1: 1}, {1: 1.1}]
        for vld_inp in valid_inputs:
            self.assertEqual(Employee(id=1, hourly_cost=1, hourly_gain=vld_inp, name='John',
                                      surname='Ally', shift_duration=8).hourly_gain, vld_inp)

        invalid_inputs = [{0: -1}, {-1: 0}, {-1: -1}]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValueError):
                Employee(id=1, hourly_cost=1.0, hourly_gain=inv_inp, name='John', surname='Ally', shift_duration=8)

        invalid_inputs = [{"1": 1}, {1: "0"}, {"1": "1"}]
        for inv_inp in invalid_inputs:
            with self.assertRaises(TypeError):
                Employee(id=1, hourly_cost=1.0, hourly_gain=inv_inp, name='John', surname='Ally', shift_duration=8)

    def test_name(self) -> None:
        invalid_inputs = [{0: -1}, [1], lambda: 0]
        for inv_inp in invalid_inputs:
            with self.assertRaises(TypeError):
                Employee(id=1, hourly_cost=1.0, hourly_gain={1: 1}, name=inv_inp, surname='Ally', shift_duration=8)

        self.assertEqual(Employee(id=1, hourly_cost=1.0, hourly_gain={1: 1}, name='John', surname='Ally',
                                  shift_duration=8).name, 'John')

    def test_surname(self) -> None:
        invalid_inputs = [{0: -1}, [1], lambda: 0]
        for inv_inp in invalid_inputs:
            with self.assertRaises(TypeError):
                Employee(id=1, hourly_cost=1.0, hourly_gain={1: 1}, name="John", surname=inv_inp, shift_duration=8)

        self.assertEqual(Employee(id=1, hourly_cost=1.0, hourly_gain={1: 1}, name='John', surname='Ally',
                                  shift_duration=8).surname, 'Ally')

    def test_shift_duration(self) -> None:
        invalid_inputs = [{0: -1}, [1], lambda: 0]
        for inv_inp in invalid_inputs:
            with self.assertRaises(TypeError):
                Employee(id=1, hourly_cost=1.0, hourly_gain={1: 1}, name="John", surname="Ally", shift_duration=inv_inp)

        invalid_inputs = [-1, -2, -3]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValueError):
                Employee(id=1, hourly_cost=1.0, hourly_gain={1: 1}, name="John", surname="Ally", shift_duration=inv_inp)

        self.assertEqual(Employee(id=1, hourly_cost=1.0, hourly_gain={1: 1}, name="John", surname="Ally",
                                  shift_duration=8).shift_duration, 8)


# noinspection PyTypeChecker
class TimeSpanTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(TimeSpanTests, self).__init__(*args, **kwargs)

    def test_fields(self) -> None:
        time_span = TimeSpan(id=1, datetime=datetime(2023, 11, 1, 12, 0, 0))

        self.assertEqual(time_span.id, 1)
        self.assertEqual(time_span.datetime, datetime(2023, 11, 1, 12, 0, 0))

    def test_id(self) -> None:
        valid_inputs = [0, 1, 2]
        for vld_inp in valid_inputs:
            self.assertEqual(TimeSpan(id=vld_inp, datetime=datetime(2023, 11, 1, 12, 0)).id, vld_inp)

        invalid_inputs = [-1.1, 1.1, 0.0]
        for inv_inp in invalid_inputs:
            with self.assertRaises(TypeError):
                TimeSpan(id=inv_inp, datetime=datetime(2023, 11, 1, 12, 0))

        invalid_inputs = [-1, -2, -3]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValueError):
                TimeSpan(id=inv_inp, datetime=datetime(2023, 11, 1, 12, 0))

    def test_datetime(self) -> None:
        valid_inputs = [datetime(2023, 11, 1, 8, 0, 0),
                        datetime(2023, 11, 1, 12, 0, 0),
                        datetime(2023, 11, 1, 16, 0, 0),
                        datetime(2023, 11, 1, 23, 0, 0)]
        for vld_inp in valid_inputs:
            self.assertEqual(TimeSpan(id=1, datetime=vld_inp).datetime, vld_inp)

        invalid_inputs = [datetime(2023, 11, 1, 6, 1, 0),
                          datetime(2023, 11, 1, 6, 0, 1),
                          datetime(2023, 11, 1, 6, 1, 1),
                          datetime(2023, 11, 1, 1, 0, 0),
                          datetime(2023, 11, 1, 2, 2, 0),
                          datetime(2023, 11, 1, 3, 0, 3),
                          datetime(2023, 11, 1, 4, 4, 4)]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValueError):
                TimeSpan(id=1, datetime=inv_inp)


class ResourceContainerTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(ResourceContainerTests, self).__init__(*args, **kwargs)

    def test_fields_defined(self) -> None:
        container = ResourceContainer()
        self.assertTrue(hasattr(container, 'machines'))
        self.assertTrue(hasattr(container, 'employees'))
        self.assertTrue(hasattr(container, 'time_span'))

    def test_fields(self) -> None:
        data = {
            "machines": (
                {"id": 0, "hourly_cost": 50.0, "hourly_gain": 120.0, "max_workers": 5, "inventory_nr": 100,
                 "demand": 1},
                {"id": 1, "hourly_cost": 45.0, "hourly_gain": 110.0, "max_workers": 7, "inventory_nr": 101,
                 "demand": 1.5}
            ),
            "employees": (
                {"id": 0, "hourly_cost": 20.0, "hourly_gain": {0: 5.0, 1: 6.0}, "name": "John", "surname": "Doe",
                 "shift_duration": 4},
                {"id": 1, "hourly_cost": 18.0, "hourly_gain": {0: 4.0, 1: 5.0}, "name": "Jane", "surname": "Smith",
                 "shift_duration": 8}
            ),
            "time_span": (
                {"id": 0, "datetime": datetime(2023, 11, 1, 6, 0, 0)},
                {"id": 1, "datetime": datetime(2023, 11, 1, 7, 0, 0)}
            )
        }

        mach = [Machine(**machine_data) for machine_data in data["machines"]]
        empl = [Employee(**employee_data) for employee_data in data["employees"]]
        time = [TimeSpan(**time_span_data) for time_span_data in data["time_span"]]

        container = ResourceContainer(machines=mach, employees=empl, time_span=time)
        self.assertEqual(container.machines, mach)
        self.assertEqual(container.employees, empl)
        self.assertEqual(container.time_span, time)

    def test_raises_validation_error(self) -> None:
        invalid_data = {
            "machines": [1], "employees": ['str'], "time_span": [1, 'str']
        }

        for key, value in invalid_data.items():
            with self.assertRaises(ValueError):
                ResourceContainer(**{key: value})


class ResourceManagerTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(ResourceManagerTests, self).__init__(*args, **kwargs)

    def test_import(self) -> None:
        imp_res: ResourceContainer | None = ResourceManager().import_resources_from_json(test_database_path)

        with open(test_database_path, 'r') as file:
            test_data = load(file)

        for empl_data in test_data["employees"]:
            empl_data["hourly_gain"] = ResourceManager.convert_keys(empl_data["hourly_gain"])
        for time_span_data in test_data["time_span"]:
            time_span_data["datetime"] = ResourceManager.convert_datetime(time_span_data["datetime"])

        test_machines = test_data['machines']
        test_employees = test_data['employees']
        test_time_span = test_data['time_span']

        for machine, test_machine_data in zip(imp_res.machines, test_machines):
            self.assertEqual(machine, Machine(id=test_machine_data['id'],
                                              hourly_cost=test_machine_data['hourly_cost'],
                                              hourly_gain=test_machine_data['hourly_gain'],
                                              max_workers=test_machine_data['max_workers'],
                                              inventory_nr=test_machine_data['inventory_nr'],
                                              demand=test_machine_data['demand']))

        for employee, test_employee_data in zip(imp_res.employees, test_employees):
            self.assertEqual(employee, Employee(id=test_employee_data['id'],
                                                hourly_cost=test_employee_data['hourly_cost'],
                                                hourly_gain=test_employee_data['hourly_gain'],
                                                name=test_employee_data['name'],
                                                surname=test_employee_data['surname'],
                                                shift_duration=test_employee_data['shift_duration']))

        for time_span, test_time_span_data in zip(imp_res.time_span, test_time_span):
            self.assertEqual(time_span, TimeSpan(id=test_time_span_data['id'],
                                                 datetime=test_time_span_data['datetime']))

    def test_validate_ids(self) -> None:
        with self.assertRaises(ResourceImportError):
            ResourceManager().import_resources_from_json(invalid_machines_database_path)
        with self.assertRaises(ResourceImportError):
            ResourceManager().import_resources_from_json(invalid_employees_database_path)
        with self.assertRaises(ResourceImportError):
            ResourceManager().import_resources_from_json(invalid_time_span_database_path)


class FactoryAssignmentScheduleTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(FactoryAssignmentScheduleTests, self).__init__(*args, **kwargs)

        self.res: ResourceContainer = ResourceManager().import_resources_from_json(test_database_path)

    def test___new___default(self) -> None:
        schedule: FactoryAssignmentSchedule = FactoryAssignmentSchedule(
            machines=self.res.machines, employees=self.res.employees, time_span=self.res.time_span,
            encountered_it=1, allowed_values=[0, 1], dtype='int32'
        )

        with open(test_database_path, 'r') as file:
            test_data = load(file)
        test_machines = test_data['machines']
        test_employees = test_data['employees']
        test_time_span = test_data['time_span']

        for empl_data in test_data["employees"]:
            empl_data["hourly_gain"] = ResourceManager.convert_keys(empl_data["hourly_gain"])
        for time_span_data in test_data["time_span"]:
            time_span_data["datetime"] = ResourceManager.convert_datetime(time_span_data["datetime"])

        for machine, test_machine_data in zip(schedule.machines, test_machines):
            self.assertEqual(machine, Machine(id=test_machine_data['id'],
                                              hourly_cost=test_machine_data['hourly_cost'],
                                              hourly_gain=test_machine_data['hourly_gain'],
                                              max_workers=test_machine_data['max_workers'],
                                              inventory_nr=test_machine_data['inventory_nr'],
                                              demand=test_machine_data['demand']))

        for employee, test_employee_data in zip(schedule.employees, test_employees):
            self.assertEqual(employee, Employee(id=test_employee_data['id'],
                                                hourly_cost=test_employee_data['hourly_cost'],
                                                hourly_gain=test_employee_data['hourly_gain'],
                                                name=test_employee_data['name'],
                                                surname=test_employee_data['surname'],
                                                shift_duration=test_employee_data['shift_duration']))

        for time_span, test_time_span_data in zip(schedule.time_span, test_time_span):
            self.assertEqual(time_span, TimeSpan(id=test_time_span_data['id'],
                                                 datetime=test_time_span_data['datetime']))
        self.assertTrue(np.all(np.ones((5, 5, 5)) == schedule))
        self.assertEqual(schedule.encountered_it, 1)
        self.assertEqual(schedule.allowed_values, [0, 1])
        self.assertEqual(schedule.dtype, 'int32')

    def test___new___input_array(self) -> None:
        schedule_default: FactoryAssignmentSchedule = FactoryAssignmentSchedule(
            machines=self.res.machines, employees=self.res.employees, time_span=self.res.time_span,
            encountered_it=1, allowed_values=[0, 1], dtype='int32'
        )

        zero_matrix: np.ndarray = np.zeros(shape=(5, 5, 5))
        schedule_template: FactoryAssignmentSchedule = FactoryAssignmentSchedule(
            input_array=zero_matrix,
            machines=self.res.machines, employees=self.res.employees, time_span=self.res.time_span,
            encountered_it=1, allowed_values=[0, 1], dtype='int32'
        )

        self.assertIsInstance(schedule_template, FactoryAssignmentSchedule)
        self.assertTrue(np.all(zero_matrix == schedule_template))
        self.assertEqual(schedule_default.machines, schedule_template.machines)
        self.assertEqual(schedule_default.employees, schedule_template.employees)
        self.assertEqual(schedule_default.time_span, schedule_template.time_span)
        self.assertEqual(schedule_default.encountered_it, schedule_template.encountered_it)
        self.assertEqual(schedule_default.allowed_values, schedule_template.allowed_values)
        self.assertEqual(schedule_default.shape, schedule_template.shape)
        self.assertEqual(schedule_default.dtype, schedule_template.dtype)

    # noinspection PyTypeChecker
    def test___factory___raising_error(self) -> None:
        with self.assertRaises(FactoryAssignmentScheduleError):
            FactoryAssignmentSchedule(machines=[1], employees=self.res.employees, time_span=self.res.time_span,
                                      allowed_values=[0, 1])

        with self.assertRaises(FactoryAssignmentScheduleError):
            FactoryAssignmentSchedule(machines=self.res.machines, employees=['str'], time_span=self.res.time_span,
                                      allowed_values=[0, 1])

        with self.assertRaises(FactoryAssignmentScheduleError):
            FactoryAssignmentSchedule(machines=self.res.machines, employees=self.res.employees, time_span=[1.1],
                                      allowed_values=[0, 1])

    def test_slice(self) -> None:
        schedule: FactoryAssignmentSchedule = FactoryAssignmentSchedule(
            machines=self.res.machines, employees=self.res.employees, time_span=self.res.time_span,
            encountered_it=1, allowed_values=[0, 1], dtype='int32'
        )

        test_slices = [
            (slice(1, 4), slice(1, 4), slice(1, 4)),  # Core region of the array
            (slice(1, 3), slice(1, 3), slice(1, 3)),  # Core region of the array
            (slice(2, 3), slice(0, 2), slice(1, 4)),  # Simple slice along each dimension
            (slice(0, 1), slice(3, 4), slice(2, 5)),  # Another simple slice
            (slice(0, 2), slice(1, 3), slice(2, 4)),  # Custom slice 1
            (slice(3, 5), slice(2, 5), slice(0, 2)),  # Custom slice 2
            (slice(0, 1), slice(0, 1), slice(0, 1)),  # Single element slice
            (slice(0, 5), slice(0, 5), slice(0, 5)),  # Entire array
        ]

        def calculate_dim(dimension_list: List[Union[Machine, Employee, TimeSpan]], dimension_slice: slice) -> int:
            """
            Function returns the dimension as int based on the params.
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

        for m_sl, e_sl, t_sl in test_slices:
            schedule_slice: FactoryAssignmentSchedule = schedule[m_sl, e_sl, t_sl]
            self.assertEqual(schedule_slice.machines, self.res.machines[m_sl])
            self.assertEqual(schedule_slice.employees, self.res.employees[e_sl])
            self.assertEqual(schedule_slice.time_span, self.res.time_span[t_sl])
            self.assertEqual(schedule_slice.encountered_it, schedule.encountered_it)
            self.assertEqual(schedule_slice.allowed_values, schedule.allowed_values)
            self.assertEqual(schedule_slice.dtype, schedule.dtype)

            valid_shape = (
                calculate_dim(dimension_list=self.res.machines, dimension_slice=m_sl),
                calculate_dim(dimension_list=self.res.employees, dimension_slice=e_sl),
                calculate_dim(dimension_list=self.res.time_span, dimension_slice=t_sl)
            )
            self.assertEqual(schedule_slice.shape, valid_shape)

    def test_view(self) -> None:
        schedule: FactoryAssignmentSchedule = FactoryAssignmentSchedule(
            machines=self.res.machines, employees=self.res.employees, time_span=self.res.time_span,
            encountered_it=1, allowed_values=[0, 1], dtype='int32'
        )

        schedule_view = schedule.view(FactoryAssignmentSchedule)

        self.assertTrue(np.all(schedule_view == schedule))
        self.assertEqual(schedule_view.machines, schedule.machines)
        self.assertEqual(schedule_view.employees, schedule.employees)
        self.assertEqual(schedule_view.time_span, schedule.time_span)
        self.assertEqual(schedule_view.encountered_it, schedule.encountered_it)
        self.assertEqual(schedule_view.allowed_values, schedule.allowed_values)
        self.assertEqual(schedule_view.dtype, schedule.dtype)
        self.assertEqual(schedule_view.shape, schedule.shape)

    # noinspection PyArgumentList
    def test_missing_constructor_params(self) -> None:
        with self.assertRaises(TypeError):
            FactoryAssignmentSchedule(
                employees=self.res.employees, time_span=self.res.time_span,
                encountered_it=1, allowed_values=[0, 1], dtype='int32'
            )

        with self.assertRaises(TypeError):
            FactoryAssignmentSchedule(
                employees=self.res.employees, time_span=self.res.time_span,
                encountered_it=1, allowed_values=[0, 1], dtype='int32'
            )

        with self.assertRaises(TypeError):
            FactoryAssignmentSchedule(
                machines=self.res.machines, employees=self.res.employees,
                encountered_it=1, allowed_values=[0, 1], dtype='int32'
            )

    def test_property_setters(self) -> None:
        schedule: FactoryAssignmentSchedule = FactoryAssignmentSchedule(
            machines=self.res.machines, employees=self.res.employees, time_span=self.res.time_span,
            encountered_it=1, allowed_values=[0, 1], dtype='int32'
        )

        with self.assertRaises(FactoryAssignmentScheduleError):
            schedule.machines = []

        with self.assertRaises(FactoryAssignmentScheduleError):
            schedule.employees = []

        with self.assertRaises(FactoryAssignmentScheduleError):
            schedule.time_span = []

        for inv_inp in [0.0, 1.0, -1, -1.1]:
            with self.assertRaises(FactoryAssignmentScheduleError):
                schedule.encountered_it = inv_inp

        for inv_inp in [0.0, 'str', 1]:
            with self.assertRaises(FactoryAssignmentScheduleError):
                schedule.allowed_values = inv_inp

    def test_allowed_values(self) -> None:
        schedule: FactoryAssignmentSchedule = FactoryAssignmentSchedule(
            machines=self.res.machines, employees=self.res.employees, time_span=self.res.time_span,
            encountered_it=1, allowed_values=[0, 1], dtype='int32'
        )

        for inv_val in [-1.1, -1, 0.1, 1.2, 'str']:
            with self.assertRaises(FactoryAssignmentScheduleError):
                schedule[0, 0, 0] = inv_val

    def test_cost(self) -> None:
        arr1 = np.array(
            [[[0], [0], [1], [0], [0]], [[0], [0], [0], [0], [0]], [[1], [0], [0], [1], [1]], [[0], [1], [0], [0], [0]],
             [[0], [0], [0], [0], [0]], ])
        arr2 = np.array([[[0, 0], [0, 0], [0, 0], [1, 0], [0, 1]], [[0, 0], [0, 0], [1, 0], [0, 0], [0, 0]],
                         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                         [[0, 1], [1, 1], [0, 0], [0, 1], [1, 0]], [[1, 0], [0, 0], [0, 0], [0, 0], [0, 0]]])
        arr3 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 1]],
                         [[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
                         [[0, 1, 1], [0, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]]])

        schedule1: FactoryAssignmentSchedule = FactoryAssignmentSchedule(input_array=arr1,
                                                                         machines=self.res.machines,
                                                                         employees=self.res.employees,
                                                                         time_span=self.res.time_span[0:1],
                                                                         encountered_it=1, allowed_values=[0, 1],
                                                                         dtype='int32'
                                                                         )

        schedule2: FactoryAssignmentSchedule = FactoryAssignmentSchedule(input_array=arr2,
                                                                         machines=self.res.machines,
                                                                         employees=self.res.employees,
                                                                         time_span=self.res.time_span[0:2],
                                                                         encountered_it=1, allowed_values=[0, 1],
                                                                         dtype='int32'
                                                                         )

        schedule3: FactoryAssignmentSchedule = FactoryAssignmentSchedule(input_array=arr3,
                                                                         machines=self.res.machines,
                                                                         employees=self.res.employees,
                                                                         time_span=self.res.time_span[0:3],
                                                                         encountered_it=1, allowed_values=[0, 1],
                                                                         dtype='int32'
                                                                         )

        self.assertEqual(schedule1.cost(), 271)
        self.assertEqual(schedule2.cost(), 503)
        self.assertEqual(schedule3.cost(), 778)

    def test_partial_assignment_not_changing_dim_list(self):
        schedule: FactoryAssignmentSchedule = FactoryAssignmentSchedule(
            machines=self.res.machines, employees=self.res.employees, time_span=self.res.time_span,
            encountered_it=1, allowed_values=[0, 1], dtype='int32'
        )

        frac_machines = self.res.machines[0:2]
        frac_employees = self.res.employees[0:2]
        frac_time_span = self.res.time_span[0:2]

        fraction: FactoryAssignmentSchedule = FactoryAssignmentSchedule(
            input_array=np.zeros((2, 2, 2)),
            machines=frac_machines, employees=frac_employees,
            time_span=frac_time_span, encountered_it=2, allowed_values=[0, 1, 2], dtype='int32'
        )

        slices = [slice(0, 2), slice(3, 5)]
        for m, e, t in product(slices, slices, slices):
            schedule[m, e, t] = fraction

        comparison_template = np.zeros((5, 5, 5))
        comparison_template[2, :, :] = np.ones((1, 1))
        comparison_template[:, 2, :] = np.ones((1, 1))
        comparison_template[:, :, 2] = np.ones((1, 1))

        self.assertTrue(np.all(schedule == comparison_template))
        self.assertEqual(schedule.shape, (5, 5, 5))
        self.assertEqual(schedule.machines, self.res.machines)
        self.assertEqual(schedule.employees, self.res.employees)
        self.assertEqual(schedule.time_span, self.res.time_span)
        self.assertEqual(schedule.encountered_it, 1)
        self.assertEqual(schedule.allowed_values, [0, 1])
        self.assertEqual(schedule.dtype, 'int32')

        self.assertEqual(fraction.shape, (2, 2, 2))
        self.assertEqual(fraction.machines, frac_machines)
        self.assertEqual(fraction.employees, frac_employees)
        self.assertEqual(fraction.time_span, frac_time_span)
        self.assertEqual(fraction.encountered_it, 2)
        self.assertEqual(fraction.allowed_values, [0, 1, 2])
        self.assertEqual(fraction.dtype, 'int32')


class UtilsFunctionTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(UtilsFunctionTests, self).__init__(*args, **kwargs)

        self.inv_prod = ResourceManager().import_resources_from_json(test_invalid_production_database_path)
        self.valid_prod = ResourceManager().import_resources_from_json(test_valid_production_database_path)

    def test_get_machine_production(self) -> None:
        shape = (len(self.valid_prod.machines), len(self.valid_prod.employees), len(self.valid_prod.time_span))
        template = np.ones(shape)
        template[[1, 0], [0, 1], :] = np.zeros(shape[2])
        production = FactoryAssignmentSchedule(
            machines=self.valid_prod.machines,
            employees=self.valid_prod.employees,
            time_span=self.valid_prod.time_span,
            allowed_values=[0, 1],
            input_array=template
        )
        (alice, bob) = production.employees
        (machine_a, machine_b) = production.machines
        proper_production = np.multiply(np.array([alice.hourly_gain[0], bob.hourly_gain[1]]),
                                        np.array([machine_a.hourly_gain, machine_b.hourly_gain])) * shape[2]
        for prod, mach in zip(proper_production, production.machines):
            self.assertEqual(get_machine_production(production, mach), prod)

    def test_get_nr_of_assigned_employees(self) -> None:
        shape = (len(self.valid_prod.machines), len(self.valid_prod.employees), len(self.valid_prod.time_span))
        template = np.ones(shape)
        template[0, 1, :] = np.zeros(shape[2])
        assignment = FactoryAssignmentSchedule(
            machines=self.valid_prod.machines,
            employees=self.valid_prod.employees,
            time_span=self.valid_prod.time_span,
            allowed_values=[0, 1],
            input_array=template
        )
        for tm_sp in assignment.time_span:
            self.assertEqual(get_nr_of_assigned_employees(assignment, assignment.machines[0], tm_sp), 1)
            self.assertEqual(get_nr_of_assigned_employees(assignment, assignment.machines[1], tm_sp), 2)

    def test_validate_machine_production(self) -> None:
        shape = (len(self.valid_prod.machines), len(self.valid_prod.employees), len(self.valid_prod.time_span))
        template = np.ones(shape)
        template[[1, 0], [0, 1], :] = np.zeros(shape[2])

        valid_production = FactoryAssignmentSchedule(
            machines=self.valid_prod.machines,
            employees=self.valid_prod.employees,
            time_span=self.valid_prod.time_span,
            allowed_values=[0, 1],
            input_array=template
        )
        invalid_production = FactoryAssignmentSchedule(
            machines=self.inv_prod.machines,
            employees=self.inv_prod.employees,
            time_span=self.inv_prod.time_span,
            allowed_values=[0, 1],
            input_array=template
        )

        self.assertFalse(validate_machine_production(schedule=invalid_production,
                                                     machine=invalid_production.machines[0]))
        self.assertFalse(validate_machine_production(schedule=invalid_production,
                                                     machine=invalid_production.machines[1]))

        self.assertTrue(validate_machine_production(schedule=valid_production, machine=valid_production.machines[0]))
        self.assertTrue(validate_machine_production(schedule=valid_production, machine=valid_production.machines[1]))

    def test_validate_total_production(self) -> None:
        shape = (len(self.valid_prod.machines), len(self.valid_prod.employees), len(self.valid_prod.time_span))
        template = np.ones(shape)
        template[[1, 0], [0, 1], :] = np.zeros(shape[2])

        valid_production = FactoryAssignmentSchedule(
            machines=self.valid_prod.machines,
            employees=self.valid_prod.employees,
            time_span=self.valid_prod.time_span,
            allowed_values=[0, 1],
            input_array=template
        )
        invalid_production = FactoryAssignmentSchedule(
            machines=self.inv_prod.machines,
            employees=self.inv_prod.employees,
            time_span=self.inv_prod.time_span,
            allowed_values=[0, 1],
            input_array=template
        )

        self.assertFalse(validate_total_production(invalid_production))
        self.assertTrue(validate_total_production(valid_production))

    def test_validate_machine_assignment(self) -> None:
        shape = (len(self.valid_prod.machines), len(self.valid_prod.employees), len(self.valid_prod.time_span))
        template = np.ones(shape)
        template[[1, 0], [0, 1], :] = np.zeros(shape[2])

        valid_production = FactoryAssignmentSchedule(
            machines=self.valid_prod.machines,
            employees=self.valid_prod.employees,
            time_span=self.valid_prod.time_span,
            allowed_values=[0, 1],
            input_array=template
        )
        invalid_production = FactoryAssignmentSchedule(
            machines=self.inv_prod.machines,
            employees=self.inv_prod.employees,
            time_span=self.inv_prod.time_span,
            allowed_values=[0, 1],
            input_array=np.ones(shape)
        )

        for mach in valid_production.machines:
            self.assertTrue(validate_machine_assignment(schedule=valid_production, machine=mach))

        for mach in invalid_production.machines:
            self.assertFalse(validate_machine_assignment(schedule=invalid_production, machine=mach))

    def test_validate_schedule_assignment(self) -> None:
        shape = (len(self.valid_prod.machines), len(self.valid_prod.employees), len(self.valid_prod.time_span))
        template = np.ones(shape)
        template[[1, 0], [0, 1], :] = np.zeros(shape[2])

        valid_production = FactoryAssignmentSchedule(
            machines=self.valid_prod.machines,
            employees=self.valid_prod.employees,
            time_span=self.valid_prod.time_span,
            allowed_values=[0, 1],
            input_array=template
        )
        invalid_production = FactoryAssignmentSchedule(
            machines=self.inv_prod.machines,
            employees=self.inv_prod.employees,
            time_span=self.inv_prod.time_span,
            allowed_values=[0, 1],
            input_array=np.ones(shape)
        )

        self.assertTrue(validate_schedule_assignment(valid_production))
        self.assertFalse(validate_schedule_assignment(invalid_production))

    def test_assign_shift(self) -> None:
        shape = (len(self.valid_prod.machines), len(self.valid_prod.employees), len(self.valid_prod.time_span))
        template = np.zeros(shape)

        assignment = FactoryAssignmentSchedule(
            machines=self.valid_prod.machines,
            employees=self.valid_prod.employees,
            time_span=self.valid_prod.time_span,
            allowed_values=[0, 1],
            input_array=template.copy()
        )

        proper_assignment = np.zeros(shape)
        proper_assignment[0, 0, 0:4] = np.ones(4)
        proper_assignment[0, 1, 4:12] = np.ones(8)
        proper_assignment[1, 0, 18:22] = np.ones(4)
        proper_assignment[1, 1, 18:26] = np.ones(8)

        (machine_1, machine_2) = assignment.machines
        (ana, bob) = assignment.employees

        assign_shift(assignment, ana, machine_1)
        assign_shift(assignment, bob, machine_1)

        assign_shift(assignment, ana, machine_2)
        assign_shift(assignment, bob, machine_2)

        self.assertTrue(np.all(proper_assignment == assignment))

        # perform this once again to test if principle: one shift per workday is present
        assign_shift(assignment, ana, machine_1)
        assign_shift(assignment, bob, machine_1)

        assign_shift(assignment, ana, machine_2)
        assign_shift(assignment, bob, machine_2)

        self.assertTrue(np.all(proper_assignment == assignment))

    def test_unassign_shift(self) -> None:
        shape = (len(self.valid_prod.machines), len(self.valid_prod.employees), len(self.valid_prod.time_span))

        proper_assignment = np.zeros(shape)
        proper_assignment[0, 0, 0:4] = np.ones(4)
        proper_assignment[0, 1, 4:12] = np.ones(8)
        proper_assignment[1, 0, 18:22] = np.ones(4)
        proper_assignment[1, 1, 18:26] = np.ones(8)

        assignment = FactoryAssignmentSchedule(
            machines=self.valid_prod.machines,
            employees=self.valid_prod.employees,
            time_span=self.valid_prod.time_span,
            allowed_values=[0, 1],
            input_array=proper_assignment.copy()
        )
        (ana, bob) = assignment.employees
        print()
        print(assignment)

        proper_assignment[1, 0, 18:22] = np.zeros(4)
        proper_assignment[1, 1, 18:26] = np.zeros(8)
        unassign_shift(assignment, ana)
        unassign_shift(assignment, bob)

        self.assertTrue(np.all(proper_assignment == assignment))

        proper_assignment[0, 0, 0:4] = np.zeros(4)
        proper_assignment[0, 1, 4:12] = np.zeros(8)
        unassign_shift(assignment, ana)
        unassign_shift(assignment, bob)

        print()
        print(assignment)

        self.assertTrue(np.all(proper_assignment == assignment))


if __name__ == "__main__":
    main()

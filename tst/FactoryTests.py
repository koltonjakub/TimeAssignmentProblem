"""This file contains all tests designed for tap_lib module from SimulatedAnnealing package."""
from copy import deepcopy
from unittest import TestCase, main
from unittest.mock import patch
from typing import List, Union
from itertools import product
from datetime import datetime
from json import load
import numpy as np
import os

from tap_lib.Factory import (
    WORK_DAY_DURATION,
    ResourceImportError, FactoryAssignmentScheduleError, ShiftAssignmentError, ShiftUnassignmentError,
    InvalidTotalProductionError, GenerateStartingSolutionError,
    Machine, Employee, TimeSpan, ResourceContainer, ResourceManager, FactoryAssignmentSchedule,
    get_machine_production, get_nr_of_assigned_employees,
    get_machine_maintenance, get_employee_salary, get_time_penalty, get_cost,
    is_valid_machine_production, is_valid_total_production, is_valid_machine_assignment, is_valid_schedule_assignment,
    assign_shift, unassign_shift, order_machines, order_employees, populate_machine_with_employee, populate_schedule,
    extend_time_span, generate_starting_solution, perform_random_sub_step, random_neighbour
)

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

test_database_path = os.path.join(parent_directory, "data", "tst_database", "test_database.json")
test_cost_database_path = os.path.join(parent_directory, "data", "tst_database", "test_cost_database.json")
test_invalid_machines_database_path = os.path.join(
    parent_directory, "data", "tst_database", "test_invalid_machine_ids_database.json")
test_invalid_employees_database_path = os.path.join(
    parent_directory, "data", "tst_database", "test_invalid_employee_ids_database.json")
test_invalid_time_span_database_path = os.path.join(
    parent_directory, "data", "tst_database", "test_invalid_time_span_ids_database.json")
test_invalid_production_database_path = os.path.join(
    parent_directory, "data", "tst_database", "test_invalid_production_database.json")
test_valid_production_database_path = os.path.join(
    parent_directory, "data", "tst_database", "test_valid_production_database.json")
test_advanced_shift_assignment_database_path = os.path.join(
    parent_directory, "data", "tst_database", "test_advanced_shift_assignment_database.json")
test_unassign_shift_database_path = os.path.join(
    parent_directory, "data", "tst_database", "test_unassign_shift_database.json")
test_populate_schedule_database_path = os.path.join(
    parent_directory, "data", "tst_database", "test_populate_schedule_database.json")
test_unable_to_create_valid_solution_database_path = os.path.join(
    parent_directory, "data", "tst_database", "test_unable_to_create_valid_solution_database.json")
test_extend_time_span_database_path = os.path.join(
    parent_directory, "data", "tst_database", "test_extend_time_span_database.json")
test_generate_starting_solution_extend_time_span_database_path = os.path.join(
    parent_directory, "data", "tst_database", "test_generate_starting_solution_extend_time_span_database.json")
test_generate_starting_solution_invalid_database_path = (
    os.path.join(parent_directory, "data", "tst_database",
                 "test_generate_starting_solution_invalid_database_database.json"))
test_perform_random_sub_step_database_path = (
    os.path.join(parent_directory, "data", "tst_database", "test_perform_random_sub_step_database.json"))
test_perform_random_sub_step_multiple_assignments_database_path = (
    os.path.join(parent_directory, "data", "tst_database",
                 "test_perform_random_sub_step_multiple_assignments_database.json"))
test_random_neighbour_database_path = os.path.join(parent_directory, "data", "tst_database",
                                                   "test_random_neighbour_database.json")

# Variables for UtilsFunctionTests.test_perform_random_sub_step_multiple_assignments parsed to @unittest.mock.patch
mock_mach1 = Machine(id=0, hourly_cost=50.0, hourly_gain=120.0, inventory_nr=100, max_workers=2, demand=1000)
mock_mach2 = Machine(id=1, hourly_cost=50.0, hourly_gain=120.0, inventory_nr=101, max_workers=2, demand=1000)
mock_ana = Employee(id=0, hourly_cost=1, hourly_gain={0: 1, 1: 1}, name='Ana', surname='Doe', shift_duration=4)
mock_bob = Employee(id=1, hourly_cost=2, hourly_gain={0: 1, 1: 1}, name='Bob', surname='Smith', shift_duration=8)


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
            self.assertEqual(Machine(id=vld_inp, hourly_cost=1.0, hourly_gain=1.0, max_workers=1, inventory_nr=123,
                                     demand=1).id, vld_inp)

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
            self.assertEqual(Machine(id=1, hourly_cost=vld_inp, hourly_gain=1.0, max_workers=1, inventory_nr=123,
                                     demand=1).hourly_cost, vld_inp)

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
        self.assertEqual(Machine(id=1, hourly_cost=1, hourly_gain=1.0, max_workers=1, inventory_nr=123,
                                 demand=1).max_workers, 1)

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
            ResourceManager().import_resources_from_json(test_invalid_machines_database_path)
        with self.assertRaises(ResourceImportError):
            ResourceManager().import_resources_from_json(test_invalid_employees_database_path)
        with self.assertRaises(ResourceImportError):
            ResourceManager().import_resources_from_json(test_invalid_time_span_database_path)


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
        self.assertTrue(np.all(np.zeros((5, 5, 5)) == schedule))
        self.assertEqual(schedule.encountered_it, 1)
        self.assertEqual(schedule.allowed_values, [0, 1])
        self.assertEqual(schedule.dtype, 'int32')
        self.assertEqual(schedule.exceeding_days, 0)

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
        self.assertEqual(schedule_default.exceeding_days, 0)

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
        self.assertEqual(schedule_view.exceeding_days, 0)

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

        for inv_inp in [0.0, '1']:
            with self.assertRaises(TypeError):
                schedule.exceeding_days = inv_inp

        for inv_inp in [-1]:
            with self.assertRaises(ValueError):
                schedule.exceeding_days = inv_inp

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
            input_array=np.ones((5, 5, 5)),
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
        self.assertEqual(schedule.exceeding_days, 0)

        self.assertEqual(fraction.shape, (2, 2, 2))
        self.assertEqual(fraction.machines, frac_machines)
        self.assertEqual(fraction.employees, frac_employees)
        self.assertEqual(fraction.time_span, frac_time_span)
        self.assertEqual(fraction.encountered_it, 2)
        self.assertEqual(fraction.allowed_values, [0, 1, 2])
        self.assertEqual(fraction.dtype, 'int32')
        self.assertEqual(fraction.exceeding_days, 0)


class UtilsFunctionTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(UtilsFunctionTests, self).__init__(*args, **kwargs)

        self.cost_db = ResourceManager().import_resources_from_json(test_cost_database_path)
        self.inv_prod = ResourceManager().import_resources_from_json(test_invalid_production_database_path)
        self.valid_prod = ResourceManager().import_resources_from_json(test_valid_production_database_path)
        self.advanced_shift = ResourceManager().import_resources_from_json(test_advanced_shift_assignment_database_path)
        self.unassign_shift = ResourceManager().import_resources_from_json(test_unassign_shift_database_path)
        self.populate_schedule = ResourceManager().import_resources_from_json(test_populate_schedule_database_path)
        self.fool_populate_schedule = ResourceManager().import_resources_from_json(
            test_unable_to_create_valid_solution_database_path)
        self.extend_time_span = ResourceManager().import_resources_from_json(test_extend_time_span_database_path)
        self.generate_schedule_with_time_span_extension = ResourceManager().import_resources_from_json(
            test_generate_starting_solution_extend_time_span_database_path
        )
        self.perform_random_sub_step = ResourceManager.import_resources_from_json(
            test_perform_random_sub_step_database_path)
        self.perform_random_sub_step_multiple_assignments = ResourceManager.import_resources_from_json(
            test_perform_random_sub_step_multiple_assignments_database_path)
        self.random_neighbour = ResourceManager.import_resources_from_json(test_random_neighbour_database_path)

    def test_machines_maintenance(self) -> None:
        shape = (len(self.cost_db.machines), len(self.cost_db.employees), len(self.cost_db.time_span))
        schedule = FactoryAssignmentSchedule(
            machines=self.cost_db.machines,
            employees=self.cost_db.employees,
            time_span=self.cost_db.time_span,
            allowed_values=[0, 1],
            input_array=np.ones(shape)
        )
        expected = [1*36, 2*36]
        for mach, exp in zip(schedule.machines, expected):
            self.assertEqual(get_machine_maintenance(schedule, mach), exp)

    def test_employee_salary(self) -> None:
        shape = (len(self.cost_db.machines), len(self.cost_db.employees), len(self.cost_db.time_span))
        schedule = FactoryAssignmentSchedule(
            machines=self.cost_db.machines,
            employees=self.cost_db.employees,
            time_span=self.cost_db.time_span,
            allowed_values=[0, 1],
            input_array=np.ones(shape)
        )
        expected = [2 * 1 * 36, 2 * 2 * 36]
        for empl, exp in zip(schedule.employees, expected):
            self.assertEqual(get_employee_salary(schedule, empl), exp)

    def test_time_penalty(self) -> None:
        shape = (len(self.cost_db.machines), len(self.cost_db.employees), len(self.cost_db.time_span))
        schedule = FactoryAssignmentSchedule(
            machines=self.cost_db.machines,
            employees=self.cost_db.employees,
            time_span=self.cost_db.time_span,
            allowed_values=[0, 1],
            input_array=np.ones(shape),
            exceeding_days=1
        )
        expected = 10*2 + 50*1**2
        self.assertEqual(get_time_penalty(schedule), expected)

    def test_get_cost(self) -> None:
        shape = (len(self.cost_db.machines), len(self.cost_db.employees), len(self.cost_db.time_span))
        schedule = FactoryAssignmentSchedule(
            machines=self.cost_db.machines,
            employees=self.cost_db.employees,
            time_span=self.cost_db.time_span,
            allowed_values=[0, 1],
            input_array=np.ones(shape),
            exceeding_days=1
        )
        expected = 1*36 + 2*36 + 2*1*36 + 2*2*36 + 10*2 + 50*1**2
        self.assertEqual(get_cost(schedule), expected)

    def test_get_machine_production(self) -> None:
        shape = (len(self.valid_prod.machines), len(self.valid_prod.employees), len(self.valid_prod.time_span))
        template = np.ones(shape)
        template[[1, 0], [0, 1], :] = np.zeros(shape[2])
        production = FactoryAssignmentSchedule(
            machines=self.valid_prod.machines,
            employees=self.valid_prod.employees,
            time_span=self.valid_prod.time_span,
            allowed_values=[0, 1],
            input_array=template.copy()
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

    def test_is_valid_machine_production(self) -> None:
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

        self.assertFalse(is_valid_machine_production(schedule=invalid_production,
                                                     machine=invalid_production.machines[0]))
        self.assertFalse(is_valid_machine_production(schedule=invalid_production,
                                                     machine=invalid_production.machines[1]))

        self.assertTrue(is_valid_machine_production(schedule=valid_production, machine=valid_production.machines[0]))
        self.assertTrue(is_valid_machine_production(schedule=valid_production, machine=valid_production.machines[1]))

    def test_is_valid_total_production(self) -> None:
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

        self.assertFalse(is_valid_total_production(invalid_production))
        self.assertTrue(is_valid_total_production(valid_production))

    def test_is_valid_machine_assignment(self) -> None:
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
            self.assertTrue(is_valid_machine_assignment(schedule=valid_production, machine=mach))

        for mach in invalid_production.machines:
            self.assertFalse(is_valid_machine_assignment(schedule=invalid_production, machine=mach))

    def test_is_valid_schedule_assignment(self) -> None:
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

        self.assertTrue(is_valid_schedule_assignment(valid_production))
        self.assertFalse(is_valid_schedule_assignment(invalid_production))

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

        (machine_1, machine_2) = assignment.machines
        (ana, bob) = assignment.employees
        proper_assignment = np.zeros(shape)

        proper_assignment[machine_1.id, ana.id, 0:ana.shift_duration] = np.ones(ana.shift_duration)
        assign_shift(assignment, ana, machine_1)
        self.assertTrue(np.all(proper_assignment == assignment))

        proper_assignment[machine_1.id, bob.id, ana.shift_duration:ana.shift_duration + bob.shift_duration] = (
            np.ones(bob.shift_duration))
        assign_shift(assignment, bob, machine_1)
        self.assertTrue(np.all(proper_assignment == assignment))

        proper_assignment[machine_2.id, ana.id, WORK_DAY_DURATION:WORK_DAY_DURATION + ana.shift_duration] = (
            np.ones(ana.shift_duration))
        assign_shift(assignment, ana, machine_2)
        self.assertTrue(np.all(proper_assignment == assignment))

        proper_assignment[machine_2.id, bob.id, WORK_DAY_DURATION:WORK_DAY_DURATION + bob.shift_duration] = (
            np.ones(bob.shift_duration))
        assign_shift(assignment, bob, machine_2)
        self.assertTrue(np.all(proper_assignment == assignment))

        # perform this once again to test if principle: one shift per workday is present
        with self.assertRaises(ShiftAssignmentError):
            assign_shift(assignment, ana, machine_1)
        with self.assertRaises(ShiftAssignmentError):
            assign_shift(assignment, bob, machine_1)

        with self.assertRaises(ShiftAssignmentError):
            assign_shift(assignment, ana, machine_2)
        with self.assertRaises(ShiftAssignmentError):
            assign_shift(assignment, bob, machine_2)

    def test_assign_shift_advanced(self) -> None:
        shape = (len(self.advanced_shift.machines),
                 len(self.advanced_shift.employees),
                 len(self.advanced_shift.time_span))

        advanced_assignment = FactoryAssignmentSchedule(
            machines=self.advanced_shift.machines,
            employees=self.advanced_shift.employees,
            time_span=self.advanced_shift.time_span,
            allowed_values=[0, 1],
            input_array=np.zeros(shape)
        )

        (machine) = advanced_assignment.machines[0]
        (ana_4h, bob_4h, cal_8h, dan_8h) = advanced_assignment.employees
        proper_assignment = np.zeros(shape)

        assign_shift(advanced_assignment, ana_4h, machine)
        proper_assignment[machine.id, ana_4h.id, 0: ana_4h.shift_duration] = np.ones(ana_4h.shift_duration)
        self.assertTrue(np.all(proper_assignment == advanced_assignment))

        assign_shift(advanced_assignment, bob_4h, machine)
        proper_assignment[machine.id, bob_4h.id, 0: bob_4h.shift_duration] = np.ones(bob_4h.shift_duration)
        self.assertTrue(np.all(proper_assignment == advanced_assignment))

        assign_shift(advanced_assignment, cal_8h, machine)
        proper_assignment[machine.id, cal_8h.id,
                          ana_4h.shift_duration: ana_4h.shift_duration + cal_8h.shift_duration] = (
            np.ones(cal_8h.shift_duration))
        self.assertTrue(np.all(proper_assignment == advanced_assignment))

        assign_shift(advanced_assignment, dan_8h, machine)
        proper_assignment[machine.id, dan_8h.id,
                          ana_4h.shift_duration: ana_4h.shift_duration + dan_8h.shift_duration] = (
            np.ones(dan_8h.shift_duration))
        self.assertTrue(np.all(proper_assignment == advanced_assignment))

        assign_shift(advanced_assignment, ana_4h, machine)
        proper_assignment[machine.id, ana_4h.id,
                          WORK_DAY_DURATION: WORK_DAY_DURATION + ana_4h.shift_duration] = (
            np.ones(ana_4h.shift_duration))
        self.assertTrue(np.all(proper_assignment == advanced_assignment))

        assign_shift(advanced_assignment, cal_8h, machine)
        proper_assignment[machine.id, cal_8h.id,
                          WORK_DAY_DURATION: WORK_DAY_DURATION + cal_8h.shift_duration] = (
            np.ones(cal_8h.shift_duration))
        self.assertTrue(np.all(proper_assignment == advanced_assignment))

        assign_shift(advanced_assignment, bob_4h, machine)
        proper_assignment[machine.id, bob_4h.id,
                          WORK_DAY_DURATION + ana_4h.shift_duration:
                          WORK_DAY_DURATION + ana_4h.shift_duration + bob_4h.shift_duration] = (
            np.ones(bob_4h.shift_duration))
        self.assertTrue(np.all(proper_assignment == advanced_assignment))

        assign_shift(advanced_assignment, dan_8h, machine)
        proper_assignment[machine.id, dan_8h.id, WORK_DAY_DURATION + cal_8h.shift_duration:
                          WORK_DAY_DURATION + cal_8h.shift_duration + dan_8h.shift_duration] = (
            np.ones(dan_8h.shift_duration))
        self.assertTrue(np.all(proper_assignment == advanced_assignment))

        assign_shift(advanced_assignment, ana_4h, machine)
        proper_assignment[machine.id, ana_4h.id, 2 * WORK_DAY_DURATION:
                          2 * WORK_DAY_DURATION + ana_4h.shift_duration] = np.ones(ana_4h.shift_duration)
        self.assertTrue(np.all(proper_assignment == advanced_assignment))

        assign_shift(advanced_assignment, cal_8h, machine)
        proper_assignment[machine.id, cal_8h.id, 2 * WORK_DAY_DURATION:
                          2 * WORK_DAY_DURATION + cal_8h.shift_duration] = (
            np.ones(cal_8h.shift_duration))
        self.assertTrue(np.all(proper_assignment == advanced_assignment))

        assign_shift(advanced_assignment, dan_8h, machine)
        proper_assignment[machine.id, dan_8h.id, 2 * WORK_DAY_DURATION + ana_4h.shift_duration:
                          2 * WORK_DAY_DURATION + ana_4h.shift_duration + dan_8h.shift_duration] = np.ones(
            dan_8h.shift_duration)
        self.assertTrue(np.all(proper_assignment == advanced_assignment))

        assign_shift(advanced_assignment, bob_4h, machine)
        proper_assignment[machine.id, bob_4h.id, 2 * WORK_DAY_DURATION + cal_8h.shift_duration:
                          2 * WORK_DAY_DURATION + cal_8h.shift_duration + bob_4h.shift_duration] = np.ones(
            bob_4h.shift_duration)
        self.assertTrue(np.all(proper_assignment == advanced_assignment))

        with self.assertRaises(ShiftAssignmentError):
            assign_shift(advanced_assignment, ana_4h, machine)
        with self.assertRaises(ShiftAssignmentError):
            assign_shift(advanced_assignment, bob_4h, machine)
        with self.assertRaises(ShiftAssignmentError):
            assign_shift(advanced_assignment, cal_8h, machine)
        with self.assertRaises(ShiftAssignmentError):
            assign_shift(advanced_assignment, dan_8h, machine)

    def test_unassign_shift(self) -> None:
        shape = (len(self.unassign_shift.machines),
                 len(self.unassign_shift.employees),
                 len(self.unassign_shift.time_span))

        (machine_1, machine_2) = self.unassign_shift.machines
        (ana, bob, cal) = self.unassign_shift.employees
        proper_assignment = np.zeros(shape)

        proper_assignment[machine_1.id, ana.id, 0: ana.shift_duration] = np.ones(ana.shift_duration)
        proper_assignment[machine_1.id, bob.id, ana.shift_duration: ana.shift_duration + bob.shift_duration] = (
            np.ones(bob.shift_duration))
        proper_assignment[machine_1.id, cal.id, ana.shift_duration + bob.shift_duration:
                          ana.shift_duration + bob.shift_duration + cal.shift_duration] = (
            np.ones(cal.shift_duration))

        proper_assignment[machine_2.id, ana.id, WORK_DAY_DURATION: WORK_DAY_DURATION + ana.shift_duration] = (
            np.ones(ana.shift_duration))
        proper_assignment[machine_2.id, bob.id, WORK_DAY_DURATION: WORK_DAY_DURATION + bob.shift_duration] = (
            np.ones(bob.shift_duration))
        proper_assignment[machine_2.id, cal.id, WORK_DAY_DURATION + ana.shift_duration:
                          WORK_DAY_DURATION + ana.shift_duration + cal.shift_duration] = (
            np.ones(cal.shift_duration))

        assignment = FactoryAssignmentSchedule(
            machines=self.unassign_shift.machines,
            employees=self.unassign_shift.employees,
            time_span=self.unassign_shift.time_span,
            allowed_values=[0, 1],
            input_array=proper_assignment.copy()
        )

        proper_assignment[machine_1.id, ana.id, 0: ana.shift_duration] = np.zeros(ana.shift_duration)
        unassign_shift(assignment, ana, machine_1)
        self.assertTrue(np.all(proper_assignment == assignment))

        proper_assignment[machine_1.id, bob.id, ana.shift_duration: ana.shift_duration + bob.shift_duration] = (
            np.zeros(bob.shift_duration))
        unassign_shift(assignment, bob, machine_1)
        self.assertTrue(np.all(proper_assignment == assignment))

        proper_assignment[machine_1.id, cal.id, ana.shift_duration + bob.shift_duration:
                          ana.shift_duration + bob.shift_duration + cal.shift_duration] = (
            np.zeros(cal.shift_duration))
        unassign_shift(assignment, cal, machine_1)
        self.assertTrue(np.all(proper_assignment == assignment))

        proper_assignment[machine_2.id, ana.id, WORK_DAY_DURATION: WORK_DAY_DURATION + ana.shift_duration] = (
            np.zeros(ana.shift_duration))
        unassign_shift(assignment, ana, machine_2)
        self.assertTrue(np.all(proper_assignment == assignment))

        proper_assignment[machine_2.id, bob.id, WORK_DAY_DURATION: WORK_DAY_DURATION + bob.shift_duration] = (
            np.zeros(bob.shift_duration))
        unassign_shift(assignment, bob, machine_2)
        self.assertTrue(np.all(proper_assignment == assignment))

        proper_assignment[machine_2.id, cal.id, WORK_DAY_DURATION + ana.shift_duration:
                          WORK_DAY_DURATION + ana.shift_duration + cal.shift_duration] = (
            np.zeros(cal.shift_duration))
        unassign_shift(assignment, cal, machine_2)
        self.assertTrue(np.all(proper_assignment == assignment))

        for machine, employee in product(self.unassign_shift.machines, self.unassign_shift.employees):
            with self.assertRaises(ShiftUnassignmentError):
                unassign_shift(schedule=assignment, employee=employee, machine=machine)

        self.assertTrue(np.all(assignment == 0))

    def test_order_machines(self) -> None:
        machines = [
            Machine(id=0, hourly_cost=1, hourly_gain=1, max_workers=1, inventory_nr="101", demand=1),
            Machine(id=1, hourly_cost=1, hourly_gain=1, max_workers=1, inventory_nr="102", demand=3),
            Machine(id=2, hourly_cost=1, hourly_gain=1, max_workers=1, inventory_nr="103", demand=4),
            Machine(id=3, hourly_cost=1, hourly_gain=1, max_workers=1, inventory_nr="104", demand=2),
            Machine(id=4, hourly_cost=1, hourly_gain=1, max_workers=1, inventory_nr="105", demand=5)
        ]
        expected = [
            Machine(id=4, hourly_cost=1, hourly_gain=1, max_workers=1, inventory_nr="105", demand=5),
            Machine(id=2, hourly_cost=1, hourly_gain=1, max_workers=1, inventory_nr="103", demand=4),
            Machine(id=1, hourly_cost=1, hourly_gain=1, max_workers=1, inventory_nr="102", demand=3),
            Machine(id=3, hourly_cost=1, hourly_gain=1, max_workers=1, inventory_nr="104", demand=2),
            Machine(id=0, hourly_cost=1, hourly_gain=1, max_workers=1, inventory_nr="101", demand=1)
        ]
        result = order_machines(machines)
        self.assertEqual(result, expected)

    def test_order_employees(self) -> None:
        employees = [
            Employee(id=0, hourly_cost=1, hourly_gain={0: 5, 1: 1}, name="", surname="", shift_duration=4),
            Employee(id=1, hourly_cost=1, hourly_gain={0: 2, 1: 2}, name="", surname="", shift_duration=4),
            Employee(id=2, hourly_cost=1, hourly_gain={0: 4, 1: 3}, name="", surname="", shift_duration=4),
            Employee(id=3, hourly_cost=1, hourly_gain={0: 1, 1: 4}, name="", surname="", shift_duration=4),
            Employee(id=4, hourly_cost=1, hourly_gain={0: 3, 1: 5}, name="", surname="", shift_duration=4)
        ]

        machine_1 = Machine(id=0, hourly_cost=1, hourly_gain=1, max_workers=1, inventory_nr="101", demand=1)
        machine_2 = Machine(id=1, hourly_cost=1, hourly_gain=1, max_workers=1, inventory_nr="102", demand=1)

        expected_1 = [
            Employee(id=0, hourly_cost=1, hourly_gain={0: 5, 1: 1}, name="", surname="", shift_duration=4),
            Employee(id=2, hourly_cost=1, hourly_gain={0: 4, 1: 3}, name="", surname="", shift_duration=4),
            Employee(id=4, hourly_cost=1, hourly_gain={0: 3, 1: 5}, name="", surname="", shift_duration=4),
            Employee(id=1, hourly_cost=1, hourly_gain={0: 2, 1: 2}, name="", surname="", shift_duration=4),
            Employee(id=3, hourly_cost=1, hourly_gain={0: 1, 1: 4}, name="", surname="", shift_duration=4)
        ]
        expected_2 = [
            Employee(id=4, hourly_cost=1, hourly_gain={0: 3, 1: 5}, name="", surname="", shift_duration=4),
            Employee(id=3, hourly_cost=1, hourly_gain={0: 1, 1: 4}, name="", surname="", shift_duration=4),
            Employee(id=2, hourly_cost=1, hourly_gain={0: 4, 1: 3}, name="", surname="", shift_duration=4),
            Employee(id=1, hourly_cost=1, hourly_gain={0: 2, 1: 2}, name="", surname="", shift_duration=4),
            Employee(id=0, hourly_cost=1, hourly_gain={0: 5, 1: 1}, name="", surname="", shift_duration=4)
        ]

        result_1 = order_employees(employees, machine_1)
        result_2 = order_employees(employees, machine_2)

        self.assertEqual(result_1, expected_1)
        self.assertEqual(result_2, expected_2)

    def test_populate_machine_with_employee(self) -> None:
        boundary_machine_1 = Machine(id=0, hourly_cost=1, hourly_gain=1, max_workers=1, inventory_nr="101", demand=40)
        boundary_machine_2 = Machine(id=1, hourly_cost=1, hourly_gain=1, max_workers=2, inventory_nr="102", demand=40)

        free_machine_1 = Machine(id=0, hourly_cost=1, hourly_gain=1, max_workers=2, inventory_nr="103", demand=1)
        free_machine_2 = Machine(id=1, hourly_cost=1, hourly_gain=1, max_workers=2, inventory_nr="104", demand=5)

        (ana, bob) = self.valid_prod.employees

        shape = (2, len(self.valid_prod.employees), len(self.valid_prod.time_span))
        boundary_schedule_1 = FactoryAssignmentSchedule(
            machines=[boundary_machine_1, boundary_machine_2],
            employees=self.valid_prod.employees,
            time_span=self.valid_prod.time_span,
            allowed_values=[0, 1],
            input_array=np.zeros(shape)
        )
        template = np.zeros(shape)
        template[boundary_machine_1.id, ana.id, 0: ana.shift_duration] = np.ones(ana.shift_duration)
        template[boundary_machine_1.id, ana.id, WORK_DAY_DURATION: WORK_DAY_DURATION + ana.shift_duration] = (
            np.ones(ana.shift_duration))
        template[boundary_machine_1.id, bob.id, ana.shift_duration: ana.shift_duration + bob.shift_duration] = (
            np.ones(bob.shift_duration))
        template[boundary_machine_1.id, bob.id, WORK_DAY_DURATION + ana.shift_duration: WORK_DAY_DURATION +
                 ana.shift_duration + bob.shift_duration] = np.ones(bob.shift_duration)
        populate_machine_with_employee(boundary_schedule_1, ana, boundary_machine_1)
        populate_machine_with_employee(boundary_schedule_1, bob, boundary_machine_1)
        populate_machine_with_employee(boundary_schedule_1, ana, boundary_machine_2)
        populate_machine_with_employee(boundary_schedule_1, bob, boundary_machine_2)
        self.assertTrue(np.all(boundary_schedule_1 == template))

        shape = (2, len(self.valid_prod.employees), len(self.valid_prod.time_span))
        boundary_schedule_2 = FactoryAssignmentSchedule(
            machines=[boundary_machine_1, boundary_machine_2],
            employees=self.valid_prod.employees,
            time_span=self.valid_prod.time_span,
            allowed_values=[0, 1],
            input_array=np.zeros(shape)
        )
        template = np.zeros(shape)
        template[boundary_machine_2.id, ana.id, 0: ana.shift_duration] = np.ones(ana.shift_duration)
        template[boundary_machine_2.id, ana.id, WORK_DAY_DURATION: WORK_DAY_DURATION + ana.shift_duration] = (
            np.ones(ana.shift_duration))
        template[boundary_machine_2.id, bob.id, 0: bob.shift_duration] = np.ones(bob.shift_duration)
        template[boundary_machine_2.id, bob.id, WORK_DAY_DURATION: WORK_DAY_DURATION + bob.shift_duration] = (
            np.ones(bob.shift_duration))
        populate_machine_with_employee(boundary_schedule_2, ana, boundary_machine_2)
        populate_machine_with_employee(boundary_schedule_2, bob, boundary_machine_2)
        populate_machine_with_employee(boundary_schedule_2, ana, boundary_machine_1)
        populate_machine_with_employee(boundary_schedule_2, bob, boundary_machine_1)
        self.assertTrue(np.all(boundary_schedule_2 == template))

        shape = (2, len(self.valid_prod.employees), len(self.valid_prod.time_span))
        free_schedule = FactoryAssignmentSchedule(
            machines=[free_machine_1, free_machine_2],
            employees=self.valid_prod.employees,
            time_span=self.valid_prod.time_span,
            allowed_values=[0, 1],
            input_array=np.zeros(shape)
        )
        template = np.zeros(shape)
        template[free_machine_1.id, bob.id, 0: bob.shift_duration] = np.ones(bob.shift_duration)
        template[free_machine_2.id, ana.id, 0: ana.shift_duration] = np.ones(ana.shift_duration)
        template[free_machine_2.id, ana.id, WORK_DAY_DURATION: WORK_DAY_DURATION + ana.shift_duration] = (
            np.ones(ana.shift_duration))
        populate_machine_with_employee(free_schedule, ana, free_machine_2)
        populate_machine_with_employee(free_schedule, bob, free_machine_1)
        populate_machine_with_employee(free_schedule, ana, free_machine_1)
        populate_machine_with_employee(free_schedule, bob, free_machine_2)
        self.assertTrue(np.all(free_schedule == template))

    def test_populate_schedule(self) -> None:
        shape = (len(self.populate_schedule.machines),
                 len(self.populate_schedule.employees),
                 len(self.populate_schedule.time_span))
        (mach1, mach2, mach3) = self.populate_schedule.machines
        (emp1, emp2, emp3, emp4) = self.populate_schedule.employees

        result = FactoryAssignmentSchedule(
            machines=self.populate_schedule.machines,
            employees=self.populate_schedule.employees,
            time_span=self.populate_schedule.time_span,
            allowed_values=[0, 1]
        )

        expected = np.zeros(shape)
        expected[mach2.id, emp1.id, 0: emp1.shift_duration] = np.ones(emp1.shift_duration)
        expected[mach2.id, emp1.id, WORK_DAY_DURATION: WORK_DAY_DURATION + emp1.shift_duration] = (
            np.ones(emp1.shift_duration))
        expected[mach1.id, emp1.id, 2*WORK_DAY_DURATION: 2*WORK_DAY_DURATION + emp1.shift_duration] = (
            np.ones(emp1.shift_duration))

        expected[mach3.id, emp4.id, 0: emp4.shift_duration] = np.ones(emp4.shift_duration)
        expected[mach3.id, emp3.id, 0: emp3.shift_duration] = np.ones(emp3.shift_duration)
        expected[mach3.id, emp2.id, emp3.shift_duration: emp3.shift_duration + emp2.shift_duration] = (
            np.ones(emp2.shift_duration))

        expected[mach3.id, emp4.id, WORK_DAY_DURATION: WORK_DAY_DURATION + emp4.shift_duration] = (
            np.ones(emp4.shift_duration))
        expected[mach3.id, emp3.id, WORK_DAY_DURATION: WORK_DAY_DURATION + emp3.shift_duration] = (
            np.ones(emp3.shift_duration))
        expected[mach3.id, emp2.id, WORK_DAY_DURATION + emp3.shift_duration: WORK_DAY_DURATION + emp3.shift_duration +
                 emp2.shift_duration] = np.ones(emp2.shift_duration)

        expected[mach3.id, emp4.id, 2*WORK_DAY_DURATION: 2*WORK_DAY_DURATION + emp4.shift_duration] = (
            np.ones(emp4.shift_duration))
        expected[mach3.id, emp3.id, 2*WORK_DAY_DURATION: 2*WORK_DAY_DURATION + emp3.shift_duration] = (
            np.ones(emp3.shift_duration))
        expected[mach3.id, emp2.id, 2*WORK_DAY_DURATION + emp3.shift_duration: 2*WORK_DAY_DURATION +
                 emp3.shift_duration + emp2.shift_duration] = np.ones(emp2.shift_duration)

        populate_schedule(result)
        self.assertTrue(np.all(expected == result))

    def test_fool_populate_schedule(self) -> None:
        """This tests performs populate_schedule function with data specially designed, so that populate_schedule
        creates an invalid schedule."""
        schedule = FactoryAssignmentSchedule(
            machines=self.fool_populate_schedule.machines,
            employees=self.fool_populate_schedule.employees,
            time_span=self.fool_populate_schedule.time_span,
            allowed_values=[0, 1]
        )
        with self.assertRaises(InvalidTotalProductionError):
            populate_schedule(schedule)

    def test_extend_time_span(self) -> None:
        expected_time_span = self.extend_time_span.time_span

        starting_period = expected_time_span[0: WORK_DAY_DURATION]
        result = extend_time_span(starting_period)
        self.assertEqual(result, expected_time_span[0: 2 * WORK_DAY_DURATION])

        starting_period = expected_time_span[0: 2 * WORK_DAY_DURATION]
        result = extend_time_span(starting_period)
        self.assertEqual(result, expected_time_span[0: 3 * WORK_DAY_DURATION])

    def test_generate_starting_solution_no_time_span_extension(self) -> None:
        expected = FactoryAssignmentSchedule(
            machines=self.populate_schedule.machines,
            employees=self.populate_schedule.employees,
            time_span=self.populate_schedule.time_span,
            allowed_values=[0, 1]
        )
        populate_schedule(expected)
        result = generate_starting_solution(test_populate_schedule_database_path)
        self.assertTrue(np.all(result == expected))
        self.assertEqual(result.exceeding_days, 0)

    def test_generate_starting_solution_with_time_span_extension(self) -> None:
        result = generate_starting_solution(test_generate_starting_solution_extend_time_span_database_path)

        self.assertTrue(is_valid_schedule_assignment(result))
        self.assertTrue(is_valid_total_production(result))
        self.assertEqual(result.exceeding_days, 5)

    def test_generate_starting_solution_invalid_database(self) -> None:
        with self.assertRaises(GenerateStartingSolutionError):
            generate_starting_solution(test_generate_starting_solution_invalid_database_path)

    @patch('random.randint', side_effect=[1, 1, 1, 0, 0, 0])
    def test_perform_random_sub_step(self, _) -> None:
        shape = (len(self.perform_random_sub_step.machines),
                 len(self.perform_random_sub_step.employees),
                 len(self.perform_random_sub_step.time_span))

        ana = self.perform_random_sub_step.employees[0]
        mach = self.perform_random_sub_step.machines[0]

        result = FactoryAssignmentSchedule(
            machines=self.perform_random_sub_step.machines,
            employees=self.perform_random_sub_step.employees,
            time_span=self.perform_random_sub_step.time_span,
            allowed_values=[0, 1],
            input_array=np.zeros(shape)
        )

        expected = np.zeros(shape)

        expected[mach.id, ana.id, 0: ana.shift_duration] = np.ones(ana.shift_duration)
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))

        expected[mach.id, ana.id, WORK_DAY_DURATION: WORK_DAY_DURATION + ana.shift_duration] = (
            np.ones(ana.shift_duration))
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))

        # further assignment not possible, should not raise an error
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))

        expected[mach.id, ana.id, WORK_DAY_DURATION: WORK_DAY_DURATION + ana.shift_duration] = (
            np.zeros(ana.shift_duration))
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))

        expected[mach.id, ana.id, 0: ana.shift_duration] = np.zeros(ana.shift_duration)
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))

        # further unassignment not possible, should not raise an error
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))

    @patch('random.randint', side_effect=[1, 1, 0, 0, 1, 1, 0, 0])
    @patch('random.choice', side_effect=[
        mock_mach1, mock_ana,  # assign
        mock_mach2, mock_bob,  # assign
        mock_mach1, mock_ana,  # unassign
        mock_mach2, mock_bob,  # unassign
        mock_mach2, mock_ana,  # assign
        mock_mach1, mock_bob,  # assign
        mock_mach2, mock_ana,  # unassign
        mock_mach1, mock_bob   # unassign
    ])
    def test_perform_random_sub_step_multiple_assignments(self, _, __) -> None:
        shape = (len(self.perform_random_sub_step_multiple_assignments.machines),
                 len(self.perform_random_sub_step_multiple_assignments.machines),
                 len(self.perform_random_sub_step_multiple_assignments.time_span))

        (ana, bob) = self.perform_random_sub_step_multiple_assignments.employees
        (mach1, mach2) = self.perform_random_sub_step_multiple_assignments.machines

        result = FactoryAssignmentSchedule(
            machines=self.perform_random_sub_step_multiple_assignments.machines,
            employees=self.perform_random_sub_step_multiple_assignments.employees,
            time_span=self.perform_random_sub_step_multiple_assignments.time_span,
            allowed_values=[0, 1],
            input_array=np.zeros(shape)
        )

        expected = np.zeros(shape)

        expected[mach1.id, ana.id, 0: ana.shift_duration] = np.ones(ana.shift_duration)
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))

        expected[mach2.id, bob.id, 0: bob.shift_duration] = np.ones(bob.shift_duration)
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))

        expected[mach1.id, ana.id, 0: ana.shift_duration] = np.zeros(ana.shift_duration)
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))
        expected[mach2.id, bob.id, 0: bob.shift_duration] = np.zeros(bob.shift_duration)
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))

        expected[mach2.id, ana.id, 0: ana.shift_duration] = np.ones(ana.shift_duration)
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))

        expected[mach1.id, bob.id, 0: bob.shift_duration] = np.ones(bob.shift_duration)
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))

        expected[mach2.id, ana.id, 0: ana.shift_duration] = np.zeros(ana.shift_duration)
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))

        expected[mach1.id, bob.id, 0: bob.shift_duration] = np.zeros(bob.shift_duration)
        perform_random_sub_step(result)
        self.assertTrue(np.all(result == expected))

    @patch('tap_lib.Factory.NEIGHBOURHOOD_DIAMETER', 1)
    def test_random_neighbour(self) -> None:
        shape = (len(self.random_neighbour.machines),
                 len(self.random_neighbour.machines),
                 len(self.random_neighbour.time_span))
        mach = self.random_neighbour.machines[0]
        ana = self.random_neighbour.employees[0]

        initial_assignment = np.zeros(shape)
        initial_assignment[mach.id, ana.id, 0: ana.shift_duration] = np.ones(ana.shift_duration)

        expected = FactoryAssignmentSchedule(
            machines=self.random_neighbour.machines,
            employees=self.random_neighbour.employees,
            time_span=self.random_neighbour.time_span,
            allowed_values=[0, 1],
            input_array=initial_assignment
        )
        original = deepcopy(expected)

        result = random_neighbour(expected)

        self.assertNotEqual(expected.cost(), result.cost())
        self.assertEqual(expected.machines, result.machines)
        self.assertEqual(expected.employees, result.employees)
        self.assertEqual(expected.time_span, result.time_span)

        # check if deepcopy of input schedule was changed and the parsed schedule was not
        self.assertTrue(np.all(original == expected))


if __name__ == "__main__":
    main()

"""This file contains all tests designed for FactoryAssignmentProblem module from SimulatedAnnealing package."""

from pydantic import ValidationError
from unittest import TestCase, main
from typing import List, Union
from itertools import product
from datetime import datetime
from json import load

from FactoryAssignmentProblem.DataTypes import (
    Resource, Machine, Employee, TimeSpan, ResourceContainer, ResourceImportError, ResourceManager,
    FactoryAssignmentSchedule, FactoryAssignmentScheduleError)

import numpy as np
import os

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
test_database_path = os.path.join(parent_directory, "data", "test_database.json")


class ResourceTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(ResourceTests, self).__init__(*args, **kwargs)

    def test_fields(self) -> None:
        resource = Resource(id=1)
        self.assertEqual(resource.id, 1)

    def test_id(self) -> None:
        valid_inputs = [0, 1, 2]
        for vld_inp in valid_inputs:
            self.assertEqual(Resource(id=vld_inp).id, vld_inp)

        invalid_inputs = [-1, -1.1, 1.1, 0.0]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValidationError):
                Resource(id=inv_inp)


class MachineTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(MachineTests, self).__init__(*args, **kwargs)

    def test_fields(self) -> None:
        machine = Machine(id=1, hourly_cost=0, hourly_gain=1.1, inventory_nr=123)

        self.assertEqual(machine.id, 1)
        self.assertEqual(machine.hourly_cost, 0)
        self.assertEqual(machine.hourly_gain, 1.1)
        self.assertEqual(machine.inventory_nr, 123)

    def test_id(self) -> None:
        valid_inputs = [0, 1, 2]
        for vld_inp in valid_inputs:
            self.assertEqual(Machine(id=vld_inp, hourly_cost=1.0, hourly_gain=1.0, inventory_nr=123).id, vld_inp)

        invalid_inputs = [-1, -1.1, 1.1, 0.0]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValidationError):
                Machine(id=inv_inp, hourly_cost=1.0, hourly_gain=1.0, inventory_nr=123)

    def test_hourly_cost(self) -> None:
        valid_inputs = [0, 0.0, 1, 1.1]
        for vld_inp in valid_inputs:
            self.assertEqual(Machine(id=1, hourly_cost=vld_inp, hourly_gain=1.0, inventory_nr=123).hourly_cost, vld_inp)

        invalid_inputs = [-1, -1.1]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValidationError):
                Machine(id=1, hourly_cost=inv_inp, hourly_gain=1.0, inventory_nr=123)

    def test_hourly_gain(self) -> None:
        valid_inputs = [0, 0.0, 1, 1.1]
        for vld_inp in valid_inputs:
            self.assertEqual(Machine(id=1, hourly_cost=1, hourly_gain=vld_inp, inventory_nr=123).hourly_gain, vld_inp)

        invalid_inputs = [-1, -1.1]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValidationError):
                Machine(id=1, hourly_cost=1.0, hourly_gain=inv_inp, inventory_nr=123)

    def test_inventory_id(self) -> None:
        valid_inputs = [0, 1]
        for vld_inp in valid_inputs:
            self.assertEqual(Machine(id=1, hourly_cost=vld_inp, hourly_gain=1.0, inventory_nr=vld_inp).inventory_nr,
                             vld_inp)

        invalid_inputs = [-1, -1.1, 0.0, 1.1]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValidationError):
                Machine(id=1, hourly_cost=1.0, hourly_gain=1.0, inventory_nr=inv_inp)


class EmployeeTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(EmployeeTests, self).__init__(*args, **kwargs)

    def test_fields(self) -> None:
        employee = Employee(id=1, hourly_cost=0, hourly_gain={1: 1}, name='John', surname='Ally')

        self.assertEqual(employee.id, 1)
        self.assertEqual(employee.hourly_cost, 0)
        self.assertEqual(employee.hourly_gain, {1: 1})
        self.assertEqual(employee.name, 'John')
        self.assertEqual(employee.surname, 'Ally')

    def test_id(self) -> None:
        valid_inputs = [0, 1, 2]
        for vld_inp in valid_inputs:
            self.assertEqual(Employee(id=vld_inp, hourly_cost=1.0, hourly_gain={1: 1.0}, name='John',
                                      surname='Ally').id, vld_inp)

        invalid_inputs = [-1, -1.1, 1.1, 0.0]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValidationError):
                Employee(id=inv_inp, hourly_cost=1.0, hourly_gain={1: 1.0}, name='John', surname='Ally')

    def test_hourly_cost(self) -> None:
        valid_inputs = [0, 0.0, 1, 1.1]
        for vld_inp in valid_inputs:
            self.assertEqual(Employee(id=1, hourly_cost=vld_inp, hourly_gain={1: 1.0}, name='John',
                                      surname='Ally').hourly_cost, vld_inp)

        invalid_inputs = [-1, -1.1]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValidationError):
                Employee(id=1, hourly_cost=inv_inp, hourly_gain={1: 1.0}, name='John', surname='Ally')

    def test_hourly_gain(self) -> None:
        valid_inputs = [{0: 1}, {0: 1.1}, {1: 1}, {1: 1.1}]
        for vld_inp in valid_inputs:
            self.assertEqual(Employee(id=1, hourly_cost=1, hourly_gain=vld_inp, name='John',
                                      surname='Ally').hourly_gain, vld_inp)

        invalid_inputs = [{0: -1}, {0: -1.1}, {-1: 1}, {-1: 1.1}, {0.1: 1}, {0.1: 1.1}, {"_": 1.0}, {None: 1.0}]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValidationError):
                Employee(id=1, hourly_cost=1.0, hourly_gain=inv_inp, name='John', surname='Ally')


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
            self.assertEqual(TimeSpan(id=vld_inp, datetime="2023-11-01T10:00:00").id, vld_inp)

        invalid_inputs = [-1, -1.1, 1.1, 0.0]
        for inv_inp in invalid_inputs:
            with self.assertRaises(ValidationError):
                TimeSpan(id=inv_inp, datetime="2023-11-01T10:00:00")

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
            with self.assertRaises(ValidationError):
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
            "machines": [
                {"id": 0, "hourly_cost": 50.0, "hourly_gain": 120.0, "inventory_nr": 100},
                {"id": 1, "hourly_cost": 45.0, "hourly_gain": 110.0, "inventory_nr": 101}
            ],
            "employees": [
                {"id": 0, "hourly_cost": 20.0, "hourly_gain": {"0": 5.0, "1": 6.0}, "name": "John", "surname": "Doe"},
                {"id": 1, "hourly_cost": 18.0, "hourly_gain": {"0": 4.0, "1": 5.0}, "name": "Jane", "surname": "Smith"}
            ],
            "time_span": [
                {"id": 0, "datetime": "2023-11-01T06:00:00"},
                {"id": 1, "datetime": "2023-11-01T07:00:00"}
            ]
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
                print(value)
                ResourceContainer(**{key: value})


class ResourceManagerTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(ResourceManagerTests, self).__init__(*args, **kwargs)

    def test_import(self) -> None:
        imp_res: ResourceContainer | None = ResourceManager().import_resources_from_json(test_database_path)

        with open(test_database_path, 'r') as file:
            test_data = load(file)
        test_machines = test_data['machines']
        test_employees = test_data['employees']
        test_time_span = test_data['time_span']

        for machine, test_machine_data in zip(imp_res.machines, test_machines):
            self.assertEqual(machine, Machine(id=test_machine_data['id'], hourly_cost=test_machine_data['hourly_cost'],
                                              hourly_gain=test_machine_data['hourly_gain'],
                                              inventory_nr=test_machine_data['inventory_nr']))

        for employee, test_employee_data in zip(imp_res.employees, test_employees):
            self.assertEqual(employee, Employee(id=test_employee_data['id'],
                                                hourly_cost=test_employee_data['hourly_cost'],
                                                hourly_gain=test_employee_data['hourly_gain'],
                                                name=test_employee_data['name'], surname=test_employee_data['surname']))

        for time_span, test_time_span_data in zip(imp_res.time_span, test_time_span):
            self.assertEqual(time_span, TimeSpan(id=test_time_span_data['id'],
                                                 datetime=test_time_span_data['datetime']))

    def test_validate_ids(self) -> None:
        imp_res = ResourceManager().import_resources_from_json(test_database_path)
        imp_res.machines[0].id = 7
        with self.assertRaises(ResourceImportError):
            ResourceManager().validate_ids(imp_res)

        imp_res = ResourceManager().import_resources_from_json(test_database_path)
        imp_res.employees[0].id = 7
        with self.assertRaises(ResourceImportError):
            ResourceManager().validate_ids(imp_res)

        imp_res = ResourceManager().import_resources_from_json(test_database_path)
        imp_res.time_span[0].id = 7
        with self.assertRaises(ResourceImportError):
            ResourceManager().validate_ids(imp_res)


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

        for machine, test_machine_data in zip(schedule.machines, test_machines):
            self.assertEqual(machine, Machine(id=test_machine_data['id'], hourly_cost=test_machine_data['hourly_cost'],
                                              hourly_gain=test_machine_data['hourly_gain'],
                                              inventory_nr=test_machine_data['inventory_nr']))

        for employee, test_employee_data in zip(schedule.employees, test_employees):
            self.assertEqual(employee, Employee(id=test_employee_data['id'],
                                                hourly_cost=test_employee_data['hourly_cost'],
                                                hourly_gain=test_employee_data['hourly_gain'],
                                                name=test_employee_data['name'], surname=test_employee_data['surname']))

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

        with self.assertRaises(FactoryAssignmentScheduleError):
            schedule.cost = 1

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
        # TODO implement proper evaluation in SimulatedAnnealing.FactoryAssignmentProblem.DataTypes.py
        pass

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


if __name__ == "__main__":
    main()

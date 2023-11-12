"""This file contains all test designed for FactoryAssignmentProblem module from SimulatedAnnealing package."""

from unittest import TestCase
from typing import List, Tuple, Dict
from SimulatedAnnealing.FactoryAssignmentProblem.DataTypes import (
    Machine, Employee, AvailableResources, ResourceManager, FactoryAssignmentSchedule, FactoryAssignmentScheduleError)


class FactoryAssignmentScheduleTest(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(FactoryAssignmentScheduleTest, self).__init__(*args, **kwargs)

        self.resource_manager = ResourceManager()
        self.machines: List[Machine] = [
            self.resource_manager.create_resource(
                hourly_cost=1, hourly_gain=1, inventory_nr=11, resource_type=AvailableResources.MACHINE)
        ]
        self.employees: List[Employee] = [
            self.resource_manager.create_resource(
                hourly_cost=1, hourly_gain=1, name='John', surname='Smith', resource_type=AvailableResources.EMPLOYEE),
            self.resource_manager.create_resource(
                hourly_cost=2, hourly_gain=2, name='Sam', surname='Allen', resource_type=AvailableResources.EMPLOYEE)
        ]
        self.time_span: List[int] = [0, 1, 2]
        self.encountered_it: int = 1
        self.allowed_values: List[int] = [0, 1]

        self.schedule: FactoryAssignmentSchedule = FactoryAssignmentSchedule(
            machines=self.machines, employees=self.employees, time_span=self.time_span,
            encountered_it=self.encountered_it, allowed_values=self.allowed_values)

    def test_new_instance_creation(self) -> None:
        self.assertEqual(self.schedule.machines, self.machines, 'machines differ')
        self.assertEqual(self.schedule.employees, self.employees, 'employees differ')
        self.assertEqual(self.schedule.time_span, self.time_span, 'time_span differ')
        self.assertEqual(self.schedule.encountered_it, self.encountered_it, 'encountered_it differs')
        self.assertEqual(self.schedule.allowed_values, self.allowed_values, 'allowed_values differ')

    def test___evaluate_cost(self) -> None:
        # TODO implement proper evaluation in SimulatedAnnealing.FactoryAssignmentProblem.DataTypes.py
        self.assertIsNone(None)

    def test_shape(self) -> None:
        self.assertEqual(self.schedule.shape, (len(self.machines), len(self.employees), len(self.time_span)),
                         msg='shape differs')


class FactoryAssignmentScheduleViewTest(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(FactoryAssignmentScheduleViewTest, self).__init__(*args, **kwargs)

        self.resource_manager = ResourceManager()
        self.machines: List[Machine] = [
            self.resource_manager.create_resource(
                hourly_cost=1, hourly_gain=1, inventory_nr=11, resource_type=AvailableResources.MACHINE),
            self.resource_manager.create_resource(
                hourly_cost=2, hourly_gain=2, inventory_nr=12, resource_type=AvailableResources.MACHINE)
        ]
        self.employees: List[Employee] = [
            self.resource_manager.create_resource(
                hourly_cost=1, hourly_gain=1, name='John', surname='Smith', resource_type=AvailableResources.EMPLOYEE),
            self.resource_manager.create_resource(
                hourly_cost=2, hourly_gain=2, name='Andrew', surname='Allen',
                resource_type=AvailableResources.EMPLOYEE),
            self.resource_manager.create_resource(
                hourly_cost=3, hourly_gain=3, name='Sam', surname='Albin', resource_type=AvailableResources.EMPLOYEE)
        ]
        self.time_span: List[int] = [0, 1, 2, 3]
        self.encountered_it: int = 1

        self.schedule: FactoryAssignmentSchedule = FactoryAssignmentSchedule(
            machines=self.machines, employees=self.employees, time_span=self.time_span,
            encountered_it=self.encountered_it)

        self.slices_ids: Dict[str: int] = {'machine': 0, 'employee': 0, 'time_span': 0}

        self.slice_machine: FactoryAssignmentSchedule = self.schedule[self.slices_ids['machine'], :, :]
        self.slice_employee: FactoryAssignmentSchedule = self.schedule[:, self.slices_ids['employee'], :]
        self.slice_time: FactoryAssignmentSchedule = self.schedule[:, :, self.slices_ids['time_span']]

        self.slices: List[FactoryAssignmentSchedule] = [self.slice_machine, self.slice_employee, self.slice_time]

    def test_machines_from_slice(self) -> None:
        # TODO add loss of dimension information test
        expected: List[List[Machine]] = [self.schedule.machines, self.schedule.machines, self.schedule.machines]

        for checked_slice, exp_res in zip(self.slices, expected):
            self.assertEqual(checked_slice.machines, exp_res, msg='machines differ')

    def test_employees_from_slice(self) -> None:
        # TODO add loss of dimension information test
        expected: List[List[Machine]] = [self.schedule.employees, self.schedule.employees, self.schedule.employees]

        for checked_slice, exp_res in zip(self.slices, expected):
            self.assertEqual(checked_slice.employees, exp_res, msg='employees differ')

    def test_time_span_from_slice(self) -> None:
        # TODO add loss of dimension information test
        expected: List[List[Machine]] = [self.schedule.time_span, self.schedule.time_span, self.schedule.time_span]

        for checked_slice, exp_res in zip(self.slices, expected):
            self.assertEqual(checked_slice.time_span, exp_res, msg='time_span differ')

    def test_encountered_it_from_slice(self) -> None:
        for checked_slice in self.slices:
            self.assertEqual(checked_slice.encountered_it, self.schedule.encountered_it, msg='encountered_it differs')

    def test_slice_shape(self) -> None:
        expected: List[Tuple[int, int,]] = [
            (len(self.employees), len(self.time_span),),
            (len(self.machines), len(self.time_span),),
            (len(self.machines), len(self.employees),)
        ]

        for checked_slice, exp_shape in zip(self.slices, expected):
            self.assertEqual(checked_slice.shape, exp_shape, msg='invalid shape of slice')


class FactoryAssignmentScheduleExceptionTest(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(FactoryAssignmentScheduleExceptionTest, self).__init__(*args, **kwargs)

        self.resource_manager = ResourceManager()
        self.machines: List[Machine] = [
            self.resource_manager.create_resource(hourly_cost=1, hourly_gain=1, inventory_nr=11,
                                                  resource_type=AvailableResources.MACHINE)
        ]
        self.employees: List[Employee] = [
            self.resource_manager.create_resource(hourly_cost=1, hourly_gain=1, name='John', surname='Smith',
                                                  resource_type=AvailableResources.EMPLOYEE)
        ]
        self.time_span: List[int] = [0]
        self.allowed_values: List[int] = [0, 1]
        self.schedule: FactoryAssignmentSchedule = FactoryAssignmentSchedule(machines=self.machines,
                                                                             employees=self.employees,
                                                                             time_span=self.time_span,
                                                                             allowed_values=self.allowed_values)

    def test___setitem__raises_error(self) -> None:
        with self.assertRaises(FactoryAssignmentScheduleError):
            self.schedule[0, 0, 0] = max(self.allowed_values) + 1

    def test___factory___raising_error(self) -> None:
        with self.assertRaises(FactoryAssignmentScheduleError):
            self.custom_array_instance = FactoryAssignmentSchedule(machines=self.machines, employees=self.employees)

        with self.assertRaises(FactoryAssignmentScheduleError):
            self.custom_array_instance = FactoryAssignmentSchedule(machines=self.machines, time_span=self.time_span)

        with self.assertRaises(FactoryAssignmentScheduleError):
            self.custom_array_instance = FactoryAssignmentSchedule(employees=self.employees, time_span=self.time_span)

    def test_machines_read_only(self) -> None:
        with self.assertRaises(FactoryAssignmentScheduleError):
            self.schedule.machines = self.machines

    def test_employees_read_only(self) -> None:
        with self.assertRaises(FactoryAssignmentScheduleError):
            self.schedule.employees = self.employees

    def test_time_span_read_only(self) -> None:
        with self.assertRaises(FactoryAssignmentScheduleError):
            self.schedule.time_span = self.time_span

    def test_cost_read_only(self) -> None:
        with self.assertRaises(FactoryAssignmentScheduleError):
            self.schedule.cost = 0

    def test_allowed_values_read_only(self) -> None:
        with self.assertRaises(FactoryAssignmentScheduleError):
            self.schedule.allowed_values = self.allowed_values

    def test_encounter_it_setter(self) -> None:
        with self.assertRaises(FactoryAssignmentScheduleError):
            self.schedule.encountered_it = 0.1

        with self.assertRaises(FactoryAssignmentScheduleError):
            self.schedule.encountered_it = -1

        with self.assertRaises(FactoryAssignmentScheduleError):
            self.schedule.encountered_it = 'str'

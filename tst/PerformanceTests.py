"""This file contains tests for performance of all functions/classes in project"""

import os
import math

from timeit import timeit
from datetime import datetime
from unittest import TestCase, main

from FactoryAssignmentProblem.DataTypes import Machine, Employee, TimeSpan, ResourceContainer
from SimulatedAnnealing.Solver import Solver


class DataTypesPerformanceTimeTests(TestCase):
    def __init__(self, *args, **kwargs):
        super(DataTypesPerformanceTimeTests, self).__init__(*args, **kwargs)

    @staticmethod
    def test_read_performance(number: int = 10**5):
        def assign_read_machine():
            machine = Machine(id=1, hourly_cost=0, hourly_gain=1.1, max_workers=1, inventory_nr=123)
            _ = machine.id
            _ = machine.hourly_cost
            _ = machine.hourly_gain
            _ = machine.max_workers
            _ = machine.inventory_nr

        def assign_read_employee():
            employee = Employee(id=1, hourly_cost=0, hourly_gain={1: 1}, name='John', surname='Ally')

            _ = employee.id
            _ = employee.hourly_cost
            _ = employee.hourly_gain
            _ = employee.name
            _ = employee.surname

        def assign_read_timespan():
            time_span = TimeSpan(id=1, datetime=datetime(2023, 11, 1, 12, 0, 0))

            _ = time_span.id
            _ = time_span.datetime

        def assign_read_resource_container():
            machine = Machine(id=1, hourly_cost=0, hourly_gain=1.1, max_workers=1, inventory_nr=123)
            employee = Employee(id=1, hourly_cost=0, hourly_gain={1: 1}, name='John', surname='Ally')
            time_span = TimeSpan(id=1, datetime=datetime(2023, 11, 1, 12, 0, 0))

            resource_container = ResourceContainer(machines=[machine], employees=[employee], time_span=[time_span])

            _ = resource_container.machines
            _ = resource_container.employees
            _ = resource_container.time_span

        results = {"Machine": timeit(assign_read_machine, number=number),
                   "Employee": timeit(assign_read_employee, number=number),
                   "TimeSpan": timeit(assign_read_timespan, number=number),
                   "ResourceContainer": timeit(assign_read_resource_container, number=number)}
        print(f'DataTypes performance tests({number} samples):')
        for key, value in results.items():
            print(f'Average __init__/read time: {key} {value/number}')

        # Results without __slots__
        # PASSED[100 %] DataTypes performance tests(100000 samples):
        # Average __init__ / read time: Machine 3.9839739999979426e-06
        # Average __init__ / read time: Employee 4.307616000000962e-06
        # Average __init__ / read time: TimeSpan 1.7587679999996908e-06
        # Average __init__ / read time: ResourceContainer 1.8110698999998933e-05


class SolverExecutionTimeTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(SolverExecutionTimeTests, self).__init__(*args, **kwargs)

    @staticmethod
    def test_simulate_annealing_execution_time() -> None:
        """
        Function to test the execution time of simulated annealing algorithm
        """
        solver = Solver(SolutionType=int, cost=lambda sol: sol, sol_gen=lambda _: 0, cool=lambda t, k: 0.1,
                        probability=lambda de, t: 0.5, init_sol=1, init_temp=10,
                        experiment_name="test_simulate_annealing")
        max_iterations = [10 ** power for power in range(1, 4 + 1)]
        execution_time = {}
        current_directory = os.getcwd()
        parent_directory = os.path.dirname(current_directory)
        logger = os.path.join(parent_directory, "logs", "test_setup_logger.log")
        solver.setup_logger(log_file_path=logger)

        number = 100
        for max_it in max_iterations:
            solver.max_iterations = max_it
            execution_time[max_it] = timeit(solver.simulate_annealing, number=number)

        spacer = " " * int(math.log10(max(max_iterations)))
        print(f'Average execution times for given max_it of plane simulated annealing algorithm({number} samples):')
        print(f'it{spacer} time')
        for key, value in execution_time.items():
            spacer = " " * int(math.log10(max(max_iterations) / key))
            print(spacer, key, value/number)


if __name__ == '__main__':
    main()

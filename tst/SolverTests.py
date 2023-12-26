"""This file contains all tests designed for Solver module from SimulatedAnnealing package."""

from tap_lib.Solver import Solver
from unittest import TestCase, main
from unittest.mock import patch
from pydantic import ValidationError

import logging as log
import numpy as np
import os


class SolverTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(SolverTests, self).__init__(*args, **kwargs)

        class Dummy:
            pass

        self.solver = Solver(SolutionType=Dummy)

    def test_proper_assignment(self) -> None:
        class Dummy:
            pass

        cost = lambda x: 1
        sol_gen = lambda x: Dummy()
        cool = lambda t, k: (1 / (k + 1)) * t
        probability = lambda d, t: np.random.rand()

        init_sol = Dummy()
        init_temp = 10
        max_iterations = 1000
        experiment_name = "dummy"
        current_directory = os.getcwd()
        parent_directory = os.path.dirname(current_directory)
        logs_directory = os.path.join(parent_directory, "logs")
        log_file_path = os.path.join(logs_directory, "tst_logs", "test_setup_logger.log")
        csv_file_path = os.path.join(logs_directory, "tst_logs", "test_csv_dump.csv")

        solver = Solver(cost=cost, sol_gen=sol_gen, cool=cool, probability=probability, init_sol=init_sol,
                        SolutionType=Dummy, init_temp=init_temp, max_iterations=max_iterations,
                        log_file_path=log_file_path, experiment_name=experiment_name, csv_file_path=csv_file_path)
        self.assertEqual(solver.cost, cost)
        self.assertEqual(solver.sol_gen, sol_gen)
        self.assertEqual(solver.cool, cool)
        self.assertEqual(solver.probability, probability)
        self.assertEqual(solver.SolutionType, Dummy)
        self.assertEqual(solver.init_sol, init_sol)
        self.assertEqual(solver.init_temp, init_temp)
        self.assertEqual(solver.log_file_path, log_file_path)
        self.assertEqual(solver.csv_file_path, csv_file_path)
        self.assertEqual(solver.max_iterations, max_iterations)
        self.assertEqual(solver.experiment_name, experiment_name)

    def test_SolutionType_raising_error(self) -> None:
        fixture = "str"
        with self.assertRaises(ValidationError):
            self.solver.SolutionType = fixture

    def test_cost_field_assignment_raising_error(self) -> None:
        fixture = "str"
        with self.assertRaises(ValidationError):
            self.solver.cost = fixture

    def test_mock_cost_field_assignment(self) -> None:
        mock_cost = lambda: 1
        with self.assertRaises(ValidationError):
            self.solver.cost = mock_cost

        mock_cost = lambda x, y: 1
        with self.assertRaises(ValidationError):
            self.solver.cost = mock_cost

        mock_cost = lambda x: "str"
        self.solver.SolutionType = int
        self.solver.cost = mock_cost
        with self.assertRaises(ValidationError):
            self.solver.init_sol = 1

        mock_cost = lambda x: []
        self.solver.SolutionType = int
        self.solver.cost = mock_cost
        with self.assertRaises(ValidationError):
            self.solver.init_sol = 1

        mock_cost = lambda x: {}
        self.solver.SolutionType = int
        self.solver.cost = mock_cost
        with self.assertRaises(ValidationError):
            self.solver.init_sol = 1

        mock_cost = lambda x: -1
        self.solver.SolutionType = int
        self.solver.cost = mock_cost
        with self.assertRaises(ValidationError):
            self.solver.init_sol = 1

        mock_cost = lambda x: 1
        self.solver.SolutionType = int
        self.solver.cost = mock_cost
        self.solver.init_sol = 1

        mock_cost = lambda x: 0
        self.solver.SolutionType = int
        self.solver.cost = mock_cost
        self.solver.init_sol = 1

    def test_sol_gen_field_assignment_raising_error(self) -> None:
        fixture = []
        with self.assertRaises(ValidationError):
            self.solver.sol_gen = fixture

    def test_mock_sol_gen_field_assignment(self) -> None:
        mock_sol_gen = lambda: 1
        with self.assertRaises(ValidationError):
            self.solver.sol_gen = mock_sol_gen

        mock_sol_gen = lambda x, y: 1
        with self.assertRaises(ValidationError):
            self.solver.sol_gen = mock_sol_gen

        mock_sol_gen = lambda x: 1
        self.solver.sol_gen = mock_sol_gen

        mock_sol_gen = lambda x: 1
        self.solver.sol_gen = None
        self.solver.SolutionType = str
        self.solver.init_sol = "str"
        with self.assertRaises(ValidationError):
            self.solver.sol_gen = mock_sol_gen

        mock_sol_gen = lambda x: "str"
        self.solver.sol_gen = None
        self.solver.SolutionType = int
        self.solver.init_sol = 1
        with self.assertRaises(ValidationError):
            self.solver.sol_gen = mock_sol_gen

        mock_sol_gen = lambda x: []
        self.solver.sol_gen = None
        self.solver.SolutionType = int
        self.solver.init_sol = 1
        with self.assertRaises(ValidationError):
            self.solver.sol_gen = mock_sol_gen

        mock_sol_gen = lambda x: {}
        self.solver.sol_gen = None
        self.solver.SolutionType = int
        self.solver.init_sol = 1
        with self.assertRaises(ValidationError):
            self.solver.sol_gen = mock_sol_gen

        class Dummy:
            pass

        mock_init_sol = Dummy()
        mock_sol_gen = lambda x: Dummy()
        self.solver.sol_gen = None
        self.solver.SolutionType = Dummy
        self.solver.init_sol = mock_init_sol
        self.solver.sol_gen = mock_sol_gen

        class OtherDummy:
            pass

        mock_sol_gen = lambda x: OtherDummy()
        with self.assertRaises(ValidationError):
            self.solver.sol_gen = mock_sol_gen

    def test_cool_field_assignment_raising_error(self) -> None:
        fixture = [1, 2, 3]
        with self.assertRaises(ValidationError):
            self.solver.cool = fixture

    def test_mock_cool_field_assignment(self) -> None:
        mock_cool = lambda: 1
        with self.assertRaises(ValidationError):
            self.solver.cool = mock_cool

        mock_cool = lambda t: 0.5 * t
        with self.assertRaises(ValidationError):
            self.solver.cool = mock_cool

        mock_cool = lambda t, k, _: 0.5 * t * (1 / k)
        with self.assertRaises(ValidationError):
            self.solver.cool = mock_cool

        mock_cool = lambda t, k: "str"
        self.solver.cool = mock_cool
        with self.assertRaises(ValidationError):
            self.solver.init_temp = 10

        mock_cool = lambda t, k: [1]
        self.solver.cool = mock_cool
        with self.assertRaises(ValidationError):
            self.solver.init_temp = 10

        mock_cool = lambda t, k: 1
        self.solver.cool = mock_cool
        self.solver.init_temp = 10

    def test_probability_field_assignment_raising_error(self) -> None:
        fixture = 1
        with self.assertRaises(ValidationError):
            self.solver.probability = fixture

    def test_mock_probability_field_assignment(self) -> None:
        mock_prob = lambda: 0.5
        with self.assertRaises(ValidationError):
            self.solver.probability = mock_prob

        mock_prob = lambda de: 0.5
        with self.assertRaises(ValidationError):
            self.solver.probability = mock_prob

        mock_prob = lambda de, t, _: 0.5
        with self.assertRaises(ValidationError):
            self.solver.probability = mock_prob

        mock_prob = lambda de, t: []
        mock_init_temp = 10
        self.solver.probability = mock_prob
        with self.assertRaises(ValidationError):
            self.solver.init_temp = mock_init_temp

        mock_prob = lambda de, t: "str"
        mock_init_temp = 10
        self.solver.probability = mock_prob
        with self.assertRaises(ValidationError):
            self.solver.init_temp = mock_init_temp

        mock_prob = lambda de, t: -1
        mock_init_temp = 10
        self.solver.probability = mock_prob
        with self.assertRaises(ValidationError):
            self.solver.init_temp = mock_init_temp

        mock_prob = lambda de, t: 1.1
        mock_init_temp = 10
        self.solver.probability = mock_prob
        with self.assertRaises(ValidationError):
            self.solver.init_temp = mock_init_temp

        mock_prob = lambda de, t: 0.5
        mock_init_temp = 10
        self.solver.probability = mock_prob
        self.solver.init_temp = mock_init_temp

    def test_init_sol_field_assignment_error(self) -> None:
        class Dummy:
            pass

        class OtherDummy:
            pass

        self.solver.SolutionType = Dummy
        fixture = OtherDummy()

        with self.assertRaises(ValidationError):
            self.solver.init_sol = fixture

        fixture = Dummy()
        self.solver.init_sol = fixture

    def test_init_temp_field_assignment_raising_error(self) -> None:
        fixture = "str"
        with self.assertRaises(ValidationError):
            self.solver.init_temp = fixture

        fixture = -1
        with self.assertRaises(ValidationError):
            self.solver.init_temp = fixture

        fixture = -1.2
        with self.assertRaises(ValidationError):
            self.solver.init_temp = fixture

        fixture = 0
        with self.assertRaises(ValidationError):
            self.solver.init_temp = fixture

        fixture = 0.0
        with self.assertRaises(ValidationError):
            self.solver.init_temp = fixture

        self.solver.init_temp = 10
        self.solver.init_temp = 10.1

    def test_max_iterations_field_assignment_raising_error(self) -> None:
        fixture = -1
        with self.assertRaises(ValidationError):
            self.solver.max_iterations = fixture

        fixture = 1.1
        with self.assertRaises(ValidationError):
            self.solver.max_iterations = fixture

    def test_log_file_path_field_assignment_raising_error(self) -> None:
        fixture = "invalid_logger.not-extension"
        with self.assertRaises(FileNotFoundError):
            self.solver.log_file_path = fixture

    def test_csv_file_field_assignment_raising_error(self) -> None:
        fixture = "_invalid_csv_"
        with self.assertRaises(ValidationError):
            self.solver.csv_file_path = fixture

    def test_log_results_field_assignment_raising_error(self) -> None:
        with self.assertRaises(ValidationError):
            self.solver.log_results = None
        self.log_results = True
        self.log_results = False

    def test_experiment_name_field_assignment_raising_error(self) -> None:
        fixture = []
        with self.assertRaises(ValidationError):
            self.solver.experiment_name = fixture

        fixture = min
        with self.assertRaises(ValidationError):
            self.solver.experiment_name = fixture

    def test_extra_fields_forbidden(self) -> None:
        with self.assertRaises(ValidationError):
            Solver(extra_fields_forbidden=[])

    def test_set_up_logger(self) -> None:
        current_directory = os.getcwd()
        parent_directory = os.path.dirname(current_directory)
        invalid_logger_directory = os.path.join(parent_directory, "invalid_logger_directory.not-extension")
        valid_logger_directory = os.path.join(parent_directory, "logs", "tst_logs", "test_setup_logger.log")
        self.assertNotEqual(valid_logger_directory, invalid_logger_directory)

        with self.assertRaises(FileNotFoundError):
            self.solver.setup_logger(invalid_logger_directory)

        self.solver.setup_logger(log_file_path=valid_logger_directory)
        self.assertTrue(len(log.root.handlers) > 0)

    def test_csv_dump(self) -> None:
        solver = Solver(SolutionType=int, cost=lambda sol: sol, sol_gen=lambda sol: 0, cool=lambda t, k: t * 0.5 ** k,
                        probability=lambda de, t: 0.5, init_sol=1, init_temp=10, max_iterations=100,
                        experiment_name="test_simulate_annealing", log_results=True)

        current_directory = os.getcwd()
        parent_directory = os.path.dirname(current_directory)
        test_csv_dump_path = os.path.join(parent_directory, "logs", "tst_logs", "test_csv_dump.csv")

        solver.csv_file_path = test_csv_dump_path

        best_sol, _ = solver.simulate_annealing()

        with open(solver.csv_file_path, "r") as file:
            lines = file.readlines()
            last_line = lines[-1].strip()
            values = last_line.split(",")
            header = lines[0].strip().split(",")
            dumped_data = dict(zip(header, values))

            self.assertEqual(dumped_data["Experiment Name"], "test_simulate_annealing")
            self.assertEqual(float(dumped_data["Initial Cost"]), 1)
            self.assertEqual(float(dumped_data["Best Cost"]), 0)
            self.assertEqual(float(dumped_data["Absolute Improvement"]), 1)
            self.assertEqual(float(dumped_data["Relative Improvement"]), 1)

        solver = Solver(SolutionType=int, cost=lambda sol: sol, sol_gen=lambda sol: sol + 1,
                        cool=lambda t, k: t * 0.5 ** k,
                        probability=lambda de, t: 0.5, init_sol=0, init_temp=10, max_iterations=100,
                        experiment_name="test_simulate_annealing", log_results=True)
        solver.csv_file_path = test_csv_dump_path

        best_sol, _ = solver.simulate_annealing()

        with open(solver.csv_file_path, "r") as file:
            lines = file.readlines()
            last_line = lines[-1].strip()
            values = last_line.split(",")
            header = lines[0].strip().split(",")
            dumped_data = dict(zip(header, values))

            self.assertEqual(dumped_data["Experiment Name"], "test_simulate_annealing")
            self.assertEqual(float(dumped_data["Initial Cost"]), 0)
            self.assertEqual(float(dumped_data["Best Cost"]), 0)
            self.assertEqual(float(dumped_data["Absolute Improvement"]), 0)
            self.assertEqual(float(dumped_data["Relative Improvement"]), 1)

    def test_simulate_annealing(self) -> None:
        solver = Solver(SolutionType=int, cost=lambda sol: sol, sol_gen=lambda sol: 0, cool=lambda t, k: t * 0.5 ** k,
                        probability=lambda de, t: 0.5, init_sol=1, init_temp=10, max_iterations=100,
                        experiment_name="test_simulate_annealing")

        best_sol, _ = solver.simulate_annealing()
        self.assertEqual(best_sol, 0)

        solver = Solver(SolutionType=int, cost=lambda sol: sol, sol_gen=lambda sol: sol + 1,
                        cool=lambda t, k: t * 0.5 ** k,
                        probability=lambda de, t: 0.5, init_sol=1, init_temp=10, max_iterations=100,
                        experiment_name="test_simulate_annealing")

        best_sol, _ = solver.simulate_annealing()
        self.assertEqual(best_sol, 1)


if __name__ == "__main__":
    main()

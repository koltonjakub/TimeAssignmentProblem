"""This file contains all the test cases for the Visualisation.py module of the Simulated Annealing Problem package."""

from unittest import TestCase, main
from tap_lib.Visualisation import Scope


class ScopeTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(ScopeTests, self).__init__(*args, **kwargs)

        class Dummy:
            pass
        self.Dummy = Dummy

    def test_field_defined(self) -> None:
        scope = Scope()
        self.assertTrue(hasattr(scope, 'iteration'))
        self.assertTrue(hasattr(scope, 'temperature'))
        self.assertTrue(hasattr(scope, 'delta_energy'))
        self.assertTrue(hasattr(scope, 'probability_of_transition'))
        self.assertTrue(hasattr(scope, 'cost_function'))
        self.assertTrue(hasattr(scope, 'best_cost_function'))
        self.assertTrue(hasattr(scope, 'visited_solution'))
        self.assertTrue(hasattr(scope, 'label'))

    def test_iteration_field_assignment_raising_error(self) -> None:
        scope = Scope()

        mock_iteration = [6.1, 'str', -2, -2.1, 0.0]
        for mock in mock_iteration:
            with self.assertRaises(ValueError):
                scope.iteration += [mock]

        scope.iteration += [1]
        self.assertEqual(scope.iteration, mock_iteration + [1])

    def test_temperature_field_assignment_raising_error(self) -> None:
        scope = Scope()

        mock_temperature = [-3, 'str', -0.1, {}, lambda: 1]
        for mock in mock_temperature:
            with self.assertRaises(ValueError):
                scope.temperature += [mock]

        scope.temperature += [1]
        self.assertEqual(scope.temperature, mock_temperature + [1])

    def test_delta_energy_field_assignment_raising_error(self) -> None:
        scope = Scope()

        mock_delta_energy = ['str', {}, lambda: int, float]
        for mock in mock_delta_energy:
            with self.assertRaises(ValueError):
                scope.delta_energy += [mock]

        scope.delta_energy += [0.3]
        self.assertEqual(scope.delta_energy, mock_delta_energy + [0.3])

    def test_probability_of_transition_field_assignment_raising_error(self) -> None:
        scope = Scope()

        mock_probability_of_transition = [1.0001, -0.3, 'str', {}, lambda: int, float]
        for mock in mock_probability_of_transition:
            with self.assertRaises(ValueError):
                scope.probability_of_transition += [mock]

        scope.probability_of_transition += [0.3]
        self.assertEqual(scope.probability_of_transition, mock_probability_of_transition + [0.3])

    def test_cost_function_field_assignment_raising_error(self) -> None:
        scope = Scope()

        mock_cost_function = [-0.3, 'str', {}, lambda: int, float, -1]
        for mock in mock_cost_function:
            with self.assertRaises(ValueError):
                scope.cost_function += [mock]

        scope.cost_function += [0.3]
        self.assertEqual(scope.cost_function, mock_cost_function + [0.3])

    def test_best_cost_function_field_assignment_raising_error(self) -> None:
        scope = Scope()

        mock_best_cost_function = [-0.3, 'str', {}, lambda: int, float, -1]
        for mock in mock_best_cost_function:
            with self.assertRaises(ValueError):
                scope.best_cost_function += [mock]

        scope.best_cost_function += [0.3]
        self.assertEqual(scope.best_cost_function, mock_best_cost_function + [0.3])


if __name__ == "__main__":
    main()

"""This file contains all the test cases for the Visualisation.py module of the Simulated Annealing Problem package."""

from unittest import TestCase, main
from pydantic import ValidationError
from SimulatedAnnealing.Visualisation.Visualisation import Scope


class ScopeTests(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(ScopeTests, self).__init__(*args, **kwargs)

        class Dummy:
            pass
        self.Dummy = Dummy

    def test_field_defined(self) -> None:
        scope = Scope()
        self.assertTrue(hasattr(scope, 'Config'))
        self.assertTrue(hasattr(scope, 'iteration'))
        self.assertTrue(hasattr(scope, 'temperature'))
        self.assertTrue(hasattr(scope, 'delta_energy'))
        self.assertTrue(hasattr(scope, 'probability_of_transition'))
        self.assertTrue(hasattr(scope, 'cost_function'))
        self.assertTrue(hasattr(scope, 'best_cost_function'))
        self.assertTrue(hasattr(scope, 'visited_solution'))
        self.assertTrue(hasattr(scope, 'label'))

    def test_Config(self) -> None:
        scope = Scope()
        self.assertTrue(scope.Config().arbitrary_types_allowed)
        self.assertTrue(scope.Config().validate_assignment)
        self.assertTrue(scope.Config().smart_union)
        self.assertEqual(scope.Config().extra, 'forbid')

    def test_iteration_field_assignment_raising_error(self) -> None:
        scope = Scope()

        mock_iteration = [0, 1, 2, 3, 4, 5, 6.1]
        with self.assertRaises(ValidationError):
            scope.iteration = mock_iteration

        mock_iteration = [0, 1, 2, 3, 'str', 5, 6]
        with self.assertRaises(ValidationError):
            scope.iteration = mock_iteration

        mock_iteration = [0, 1, -2, 3, 4, 5, 6]
        with self.assertRaises(ValidationError):
            scope.iteration = mock_iteration

        mock_iteration = [0, 1, 2, 3, 4, 5, 6]
        scope.iteration = mock_iteration
        self.assertEqual(scope.iteration, mock_iteration)

        with self.assertRaises(ValidationError):
            scope.iteration += [1.1]

    def test_temperature_field_assignment_raising_error(self) -> None:
        scope = Scope()

        mock_temperature = [10, 9, 8, 7, -3, 2, 1]
        with self.assertRaises(ValidationError):
            scope.temperature = mock_temperature

        mock_temperature = [10, 9, 8, -0.1, 3, 2, 1]
        with self.assertRaises(ValidationError):
            scope.temperature = mock_temperature

        mock_temperature = [10, 9, 'str', 7, 3, 2, 1]
        with self.assertRaises(ValidationError):
            scope.temperature = mock_temperature

        mock_temperature = [10, 9, 8, 7, 3, 2, 1]
        scope.temperature = mock_temperature
        self.assertEqual(scope.temperature, mock_temperature)

        with self.assertRaises(ValidationError):
            scope.temperature += [-0.1]

    def test_delta_energy_field_assignment_raising_error(self) -> None:
        scope = Scope()

        mock_delta_energy = [10, 9, 8, 7, [1], 2, 1]
        with self.assertRaises(ValidationError):
            scope.delta_energy = mock_delta_energy

        mock_delta_energy = [10, 9, 'str', 7, 3, 2, 1]
        with self.assertRaises(ValidationError):
            scope.delta_energy = mock_delta_energy

        mock_delta_energy = [10, 9, 8, 7, 3, 2, 1]
        scope.delta_energy = mock_delta_energy
        self.assertEqual(scope.delta_energy, mock_delta_energy)

        with self.assertRaises(ValidationError):
            scope.delta_energy += ['str']

    def test_probability_of_transition_field_assignment_raising_error(self) -> None:
        scope = Scope()

        mock_probability_of_transition = [0.1, 0.2, 0.3, 1.0001]
        with self.assertRaises(ValidationError):
            scope.probability_of_transition = mock_probability_of_transition

        mock_probability_of_transition = [0.1, 0.2, -0.3, 0.4]
        with self.assertRaises(ValidationError):
            scope.probability_of_transition = mock_probability_of_transition

        mock_probability_of_transition = [0.1, 0.2, 'str', 0.4]
        with self.assertRaises(ValidationError):
            scope.probability_of_transition = mock_probability_of_transition

        mock_probability_of_transition = []
        scope.probability_of_transition = mock_probability_of_transition
        self.assertEqual(scope.probability_of_transition, mock_probability_of_transition)

        with self.assertRaises(ValidationError):
            scope.probability_of_transition += [-0.1]

    def test_cost_function_field_assignment_raising_error(self) -> None:
        scope = Scope()

        mock_cost_function = [0.1, 0.2, -1, 1.0001]
        with self.assertRaises(ValidationError):
            scope.cost_function = mock_cost_function

        mock_cost_function = [0.1, 0.2, -0.3, 0.4]
        with self.assertRaises(ValidationError):
            scope.cost_function = mock_cost_function

        mock_cost_function = ['0.1', 0.2, 0.3, 0.4]
        scope.cost_function = mock_cost_function
        self.assertEqual(scope.cost_function, [0.1] + mock_cost_function[1::])

        with self.assertRaises(ValidationError):
            scope.cost_function += ['str']

    def test_best_cost_function_field_assignment_raising_error(self) -> None:
        scope = Scope()

        mock_best_cost_function = [0.1, 0.2, -1, 1.0001]
        with self.assertRaises(ValidationError):
            scope.best_cost_function = mock_best_cost_function

        mock_best_cost_function = [0.1, 0.2, -0.3, 0.4]
        with self.assertRaises(ValidationError):
            scope.best_cost_function = mock_best_cost_function

        mock_best_cost_function = [0.1, 0.2, 0.3, 0.4]
        scope.best_cost_function = mock_best_cost_function
        with self.assertRaises(ValidationError):
            scope.best_cost_function += ['str']

    def test_extra_field_assignment_raising_error(self) -> None:
        scope = Scope()
        with self.assertRaises(ValueError):
            scope.dummy_field = self.Dummy()


if __name__ == "__main__":
    main()

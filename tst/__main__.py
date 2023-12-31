import unittest
import time

from SolverTests import SolverTests
from VisualisationTests import ScopeTests
from FactoryTests import (MachineTests, EmployeeTests, TimeSpanTests,
                          ResourceContainerTests, ResourceManagerTests,
                          FactoryAssignmentScheduleTests, UtilsFunctionTests)

if __name__ == '__main__':
    start_time = time.time()

    test_suite = unittest.TestSuite()

    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(SolverTests))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(ScopeTests))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(MachineTests))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(EmployeeTests))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TimeSpanTests))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(ResourceContainerTests))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(ResourceManagerTests))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(FactoryAssignmentScheduleTests))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(UtilsFunctionTests))

    test_result = unittest.TestResult()
    test_suite.run(result=test_result)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Number of tests run:", test_result.testsRun)
    print("Number of errors:", len(test_result.errors))
    print("Number of failures:", len(test_result.failures))
    print("Number of skipped tests:", len(test_result.skipped))
    print("Test successful:", test_result.wasSuccessful())
    print("Time taken for tests:", elapsed_time, "seconds")
    print()
    print(f'Remembered tests:')
    print(f'Errors: {test_result.errors}')
    print(f'Failures: {test_result.failures}')
    print(f'Skipped: {test_result.skipped}')

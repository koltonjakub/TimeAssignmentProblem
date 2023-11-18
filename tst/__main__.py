import unittest
from FactoryAssignmentProblemTests import (MachineTests, EmployeeTests, TimeSpanTests,
                                           ResourceContainerTests, ResourceManagerTests, FactoryAssignmentScheduleTests)
from SolverTests import SolverTests

if __name__ == '__main__':
    test_suite = unittest.TestSuite()

    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(MachineTests))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(EmployeeTests))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TimeSpanTests))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(ResourceContainerTests))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(ResourceManagerTests))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(FactoryAssignmentScheduleTests))

    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(SolverTests))

    test_result = unittest.TestResult()
    test_suite.run(result=test_result)

    print("Number of tests run:", test_result.testsRun)
    print("Number of errors:", len(test_result.errors))
    print("Number of failures:", len(test_result.failures))
    print("Number of skipped tests:", len(test_result.skipped))
    print("Test successful:", test_result.wasSuccessful())

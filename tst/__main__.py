import unittest
from FactoryAssignmentProblemTests import (ResourceTest, MachineTest, EmployeeTest, TimeSpanTest,
                                           ResourceContainerTest, ResourceManagerTest, FactoryAssignmentScheduleTest)

if __name__ == '__main__':
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(ResourceTest))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(MachineTest))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(EmployeeTest))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TimeSpanTest))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(ResourceContainerTest))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(ResourceManagerTest))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(FactoryAssignmentScheduleTest))

    test_result = unittest.TestResult()
    test_suite.run(result=test_result)

    print("Number of tests run:", test_result.testsRun)
    print("Number of errors:", len(test_result.errors))
    print("Number of failures:", len(test_result.failures))
    print("Number of skipped tests:", len(test_result.skipped))
    print("Test successful:", test_result.wasSuccessful())

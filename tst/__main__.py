import unittest
from FactoryAssignmentProblemTests import FactoryAssignmentScheduleViewTest, FactoryAssignmentScheduleExceptionTest

if __name__ == '__main__':
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(FactoryAssignmentScheduleViewTest))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(FactoryAssignmentScheduleExceptionTest))

    test_result = unittest.TestResult()
    test_suite.run(result=test_result)

    print("Number of tests run:", test_result.testsRun)
    print("Number of errors:", len(test_result.errors))
    print("Number of failures:", len(test_result.failures))
    print("Number of skipped tests:", len(test_result.skipped))
    print("Number of successful tests:", test_result.wasSuccessful())

"""
================================================================================
Master Test Suite for All Project Phases (test_all_phases.py)
================================================================================

Aggregates and executes unit and integration tests across all completed project phases.

Covered Phases:
    - Phase 2: Corpus Ingestion & Corpus-Graph Construction
    - Phase 3: Stage 1 Initial Retrieval Baselines & TREC Evaluation
"""

import sys
import unittest

from tests.test_phase2 import TestPhase2Implementation
from tests.test_phase3 import TestPhase3Implementation


def suite():
    """Create master test suite combining all phase tests."""
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    test_suite.addTests(loader.loadTestsFromTestCase(TestPhase2Implementation))
    test_suite.addTests(loader.loadTestsFromTestCase(TestPhase3Implementation))
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    if not result.wasSuccessful():
        sys.exit(1)

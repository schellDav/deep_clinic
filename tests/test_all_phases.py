"""
================================================================================
Master Test Suite for All Project Phases (test_all_phases.py)
================================================================================

Aggregates and executes unit and integration tests across all completed project phases.

Covered Phases:
    - Phase 2: Corpus Ingestion & Corpus-Graph Construction
    - Phase 3: Stage 1 Initial Retrieval Baselines & TREC Evaluation
    - Phase 4: Stage 2 Cross-Encoder & Stage 3 GAR Candidate Expansion
"""

import sys
import unittest

from tests.test_phase2 import TestPhase2Implementation
from tests.test_phase3 import TestPhase3Implementation
from tests.test_phase4 import TestPhase4Implementation
from tests.test_phase5 import TestPhase5Implementation


def suite():
    """Create master test suite combining all phase tests."""
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    test_suite.addTests(loader.loadTestsFromTestCase(TestPhase2Implementation))
    test_suite.addTests(loader.loadTestsFromTestCase(TestPhase3Implementation))
    test_suite.addTests(loader.loadTestsFromTestCase(TestPhase4Implementation))
    test_suite.addTests(loader.loadTestsFromTestCase(TestPhase5Implementation))
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    if not result.wasSuccessful():
        sys.exit(1)

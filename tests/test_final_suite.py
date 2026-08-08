import unittest
import sys
from pathlib import Path

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_phase1_ux_boot import TestPhase1UXBoot
from tests.test_phase2_preflight import TestPhase2Preflight
from tests.test_phase3_resilience import TestPhase3Resilience
from tests.test_phase4_lstm_opt import TestPhase4LSTMOpt
from tests.test_phase5_strict_validation import TestPhase5StrictValidation

def suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1UXBoot))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2Preflight))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3Resilience))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase4LSTMOpt))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase5StrictValidation))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    if not result.wasSuccessful():
        sys.exit(1)

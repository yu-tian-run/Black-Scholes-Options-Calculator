import unittest
from engine import OptionInputs, OptionEngine

class TestOptionEngine(unittest.TestCase):
    def setUp(self):
        # Standard ATM Call test case
        self.inputs = OptionInputs(
            spot_price=100.0, strike_price=100.0, time_to_expiry=1.0,
            volatility=0.2, risk_free_rate=0.05, option_type='call'
        )
        self.engine = OptionEngine(self.inputs)

    def test_price_sanity(self):
        price = self.engine.calc_price()
        self.assertGreater(price, 0)
        self.assertLess(price, 100)

    def test_delta_bounds(self):
        greeks = self.engine.calc_greeks()
        self.assertGreaterEqual(greeks['delta'], 0)
        self.assertLessEqual(greeks['delta'], 1.0)

if __name__ == '__main__':
    unittest.main()

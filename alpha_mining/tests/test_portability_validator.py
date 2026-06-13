from __future__ import annotations

import unittest

from alpha_mining.validators import validate_portability


class TestPortabilityValidator(unittest.TestCase):
    def test_validator(self) -> None:
        good = validate_portability("rank(close)", max_operator_count=4, max_field_count=2)
        bad = validate_portability(
            "rank(ts_mean(close + volume + open + high, 5))",
            max_operator_count=2,
            max_field_count=2,
        )
        self.assertTrue(good.is_valid)
        self.assertFalse(bad.is_valid)


if __name__ == "__main__":
    unittest.main()

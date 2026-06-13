from __future__ import annotations

import unittest

from alpha_mining.mining.expand import expand_template
from alpha_mining.schema import AlphaTemplate


class TestTemplateExpand(unittest.TestCase):
    def test_expand(self) -> None:
        tpl = AlphaTemplate(template_id="t1", family="single_ts", expression="ts_mean({field}, {d})")
        out = expand_template(tpl, {"field": ["close"], "d": [5, 10]})
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()

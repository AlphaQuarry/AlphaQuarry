from __future__ import annotations

import unittest

from alpha_mining.mining.search import build_search_space
from alpha_mining.schema import AlphaTemplate


class TestSearchFilters(unittest.TestCase):
    def test_skip_group_template_when_group_missing(self) -> None:
        templates = [
            AlphaTemplate(
                template_id="tpl_group",
                family="single_group",
                expression="group_rank({field}, {group})",
                placeholders={"field": ["close"], "group": ["industry"]},
            )
        ]
        out = build_search_space(
            templates=templates,
            pools={},
            include_families={"single_group"},
            available_fields={"close"},
            available_groups=set(),
            skip_templates_with_missing_group=True,
        )
        self.assertEqual(out, [])

    def test_keep_group_template_when_group_available(self) -> None:
        templates = [
            AlphaTemplate(
                template_id="tpl_group",
                family="single_group",
                expression="group_rank({field}, {group})",
                placeholders={"field": ["close"], "group": ["industry"]},
            )
        ]
        out = build_search_space(
            templates=templates,
            pools={},
            include_families={"single_group"},
            available_fields={"close"},
            available_groups={"industry"},
            skip_templates_with_missing_group=True,
        )
        self.assertEqual(out, [("tpl_group", "group_rank(close, industry)")])


if __name__ == "__main__":
    unittest.main()

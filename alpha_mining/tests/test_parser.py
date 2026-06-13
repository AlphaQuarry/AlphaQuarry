from __future__ import annotations

import unittest
import warnings

from alpha_mining.ast_nodes import LiteralNode
from alpha_mining.parser import parse_expression


class TestParser(unittest.TestCase):
    def test_basic_parse(self) -> None:
        node = parse_expression("ts_mean(close, 5)")
        self.assertIsNotNone(node)

    def test_literals_parse_without_deprecated_ast_node_warnings(self) -> None:
        parse_expression.cache_clear()

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            node = parse_expression("1")

        self.assertIsInstance(node, LiteralNode)
        self.assertEqual(node.value, 1)


if __name__ == "__main__":
    unittest.main()

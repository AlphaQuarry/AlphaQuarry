from __future__ import annotations

import sys

from factor_research.utils import get_logger


def test_factor_research_default_logger_writes_to_stdout() -> None:
    logger = get_logger("factor_research_test_stdout")
    existing_handlers = list(logger.handlers)
    for handler in existing_handlers:
        logger.removeHandler(handler)
    try:
        configured = get_logger("factor_research_test_stdout")
        assert any(getattr(handler, "stream", None) is sys.stdout for handler in configured.handlers)
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        for handler in existing_handlers:
            logger.addHandler(handler)

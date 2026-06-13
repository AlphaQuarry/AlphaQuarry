"""Compatibility entrypoint for legacy notebook imports.

This file re-exports all public functions from the refactored package so that
`from factor_alalyze_lib import *` continues to work.
"""

from factor_research import *  # noqa: F401,F403

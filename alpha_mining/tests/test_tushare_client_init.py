from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from alpha_mining.datasource.config import TushareSettings
from alpha_mining.datasource.tushare_client import TushareClient


class _FakeDataApi:
    def __init__(self) -> None:
        self._DataApi__token = ""
        self._DataApi__http_url = ""


class TestTushareClientInit(unittest.TestCase):
    def test_init_standard_user_no_custom_endpoint(self) -> None:
        """Standard Tushare users (no http_url) should NOT touch DataApi internals."""
        fake_ts = types.SimpleNamespace()
        fake_api = _FakeDataApi()

        fake_ts.set_token = lambda _token: None
        fake_ts.pro_api = lambda _token: fake_api

        with patch.dict("sys.modules", {"tushare": fake_ts}):
            client = TushareClient(TushareSettings(token="abc123"))

        self.assertIsNotNone(client)
        # pro_api already sets the token internally; __init__ must not overwrite it.
        self.assertEqual(fake_api._DataApi__token, "")
        self.assertEqual(fake_api._DataApi__http_url, "")

    def test_init_custom_endpoint_overrides_internals(self) -> None:
        """Private deployments with a custom http_url override DataApi internals."""
        fake_ts = types.SimpleNamespace()
        fake_api = _FakeDataApi()

        fake_ts.set_token = lambda _token: None
        fake_ts.pro_api = lambda _token: fake_api

        with patch.dict("sys.modules", {"tushare": fake_ts}):
            client = TushareClient(
                TushareSettings(
                    token="abc123",
                    http_url="https://custom-api.example.com",
                )
            )

        self.assertIsNotNone(client)
        self.assertEqual(fake_api._DataApi__token, "abc123")
        self.assertEqual(fake_api._DataApi__http_url, "https://custom-api.example.com")

    def test_init_tolerates_set_token_failure(self) -> None:
        """set_token can fail in locked environments; __init__ should still succeed."""
        fake_ts = types.SimpleNamespace()
        fake_api = _FakeDataApi()

        def _set_token(_token: str) -> None:
            raise PermissionError("locked home")

        fake_ts.set_token = _set_token
        fake_ts.pro_api = lambda _token: fake_api

        with patch.dict("sys.modules", {"tushare": fake_ts}):
            client = TushareClient(TushareSettings(token="abc123"))

        self.assertIsNotNone(client)


if __name__ == "__main__":
    unittest.main()

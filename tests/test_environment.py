import os
import unittest
from unittest.mock import patch

from efi_pilot.utils.environment import require_api_keys


class EnvironmentTests(unittest.TestCase):
    def test_returns_only_requested_keys(self):
        values = {
            "DEEPSEEK_API_KEY": "deepseek-test",
            "BOCHA_API_KEY": "bocha-test",
            "QWEN_API_KEY": "qwen-test",
        }
        with patch.dict(os.environ, values, clear=True):
            self.assertEqual(
                require_api_keys("qwen"),
                {"qwen": "qwen-test"},
            )

    def test_reports_all_missing_variables_without_values(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "DEEPSEEK_API_KEY, BOCHA_API_KEY, QWEN_API_KEY",
            ):
                require_api_keys("deepseek", "bocha", "qwen")

    def test_rejects_unknown_service(self):
        with self.assertRaisesRegex(ValueError, "Unknown API service"):
            require_api_keys("unknown")


if __name__ == "__main__":
    unittest.main()

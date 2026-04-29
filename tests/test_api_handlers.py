from __future__ import annotations

import unittest

from gaoagent.api.api_handlers import ApiHandlers


class TestPickFirstModel(unittest.TestCase):
    def test_valid_dict(self) -> None:
        api_body = {"models": {"gpt-4": {}, "gpt-3.5": {}}}
        result = ApiHandlers()._pick_first_model(api_body)
        self.assertIn(result, ["gpt-4", "gpt-3.5"])

    def test_empty_models(self) -> None:
        result = ApiHandlers()._pick_first_model({"models": {}})
        self.assertEqual(result, "")

    def test_non_dict_body(self) -> None:
        result = ApiHandlers()._pick_first_model("not a dict")
        self.assertEqual(result, "")

    def test_none_body(self) -> None:
        result = ApiHandlers()._pick_first_model(None)
        self.assertEqual(result, "")

    def test_no_models_key(self) -> None:
        result = ApiHandlers()._pick_first_model({"other": "data"})
        self.assertEqual(result, "")

    def test_models_not_dict(self) -> None:
        result = ApiHandlers()._pick_first_model({"models": "not a dict"})
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()

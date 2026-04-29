from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from gaoagent.core.CoreConfigDefault import CoreConfigDefault


class TestValidateMcpConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = CoreConfigDefault()

    def test_valid_stdio_config(self) -> None:
        config = {"my-server": {"type": "stdio", "command": "python", "args": ["-m", "server"], "timeout": 30, "disabled": False}}
        self.validator._validate_mcp_config(config)

    def test_valid_sse_config(self) -> None:
        config = {"my-sse": {"type": "sse", "url": "http://localhost:8080", "timeout": 30, "disabled": False}}
        self.validator._validate_mcp_config(config)

    def test_valid_streamable_http_config(self) -> None:
        config = {"my-http": {"type": "streamable_http", "url": "http://localhost:8080", "timeout": 30, "disabled": False}}
        self.validator._validate_mcp_config(config)

    def test_not_dict_raises(self) -> None:
        with self.assertRaises(Exception):
            self.validator._validate_mcp_config("not a dict")

    def test_empty_dict_raises(self) -> None:
        with self.assertRaises(Exception):
            self.validator._validate_mcp_config({})

    def test_multiple_keys_raises(self) -> None:
        config = {"a": {"type": "stdio", "command": "x"}, "b": {"type": "stdio", "command": "y"}}
        with self.assertRaises(Exception):
            self.validator._validate_mcp_config(config)

    def test_stdio_missing_command_raises(self) -> None:
        config = {"my-server": {"type": "stdio"}}
        with self.assertRaises(Exception):
            self.validator._validate_mcp_config(config)

    def test_sse_missing_url_raises(self) -> None:
        config = {"my-sse": {"type": "sse"}}
        with self.assertRaises(Exception):
            self.validator._validate_mcp_config(config)

    def test_body_not_dict_raises(self) -> None:
        config = {"my-server": "not a dict"}
        with self.assertRaises(Exception):
            self.validator._validate_mcp_config(config)

    def test_invalid_type_raises(self) -> None:
        config = {"my-server": {"type": "invalid_type", "command": "x"}}
        with self.assertRaises(Exception):
            self.validator._validate_mcp_config(config)


class TestReadJson(unittest.TestCase):
    def test_nonexistent_file(self) -> None:
        result = CoreConfigDefault()._read_json(Path("/nonexistent/file.json"))
        self.assertIsNone(result)

    @patch("gaoagent.core.CoreConfigDefault.Console")
    def test_invalid_json(self, mock_console) -> None:
        import tempfile
        mock_console.confirm.return_value = True
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            f.flush()
            result = CoreConfigDefault()._read_json(Path(f.name))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

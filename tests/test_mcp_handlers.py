from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gaoagent.mcp.mcp_handlers import MCPHandlers


class TestMCPStatusOf(unittest.TestCase):
    def test_enabled(self) -> None:
        h = MCPHandlers()
        self.assertEqual(h._status_of({"disabled": False}), "enabled")

    def test_disabled(self) -> None:
        h = MCPHandlers()
        self.assertEqual(h._status_of({"disabled": True}), "disabled")

    def test_no_disabled_key(self) -> None:
        h = MCPHandlers()
        self.assertEqual(h._status_of({"command": "x"}), "enabled")

    def test_non_dict(self) -> None:
        h = MCPHandlers()
        self.assertEqual(h._status_of("not a dict"), "invalid")

    def test_none(self) -> None:
        h = MCPHandlers()
        self.assertEqual(h._status_of(None), "invalid")


class TestMCPLoadMcpServers(unittest.TestCase):
    def test_valid_config(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"mcpServers": {"srv1": {"type": "stdio"}}}, f)
            f.flush()
            h = MCPHandlers()
            result = h._load_mcp_servers(Path(f.name))
            self.assertIn("srv1", result)

    def test_missing_file(self) -> None:
        h = MCPHandlers()
        result = h._load_mcp_servers(Path("/nonexistent/file.json"))
        self.assertEqual(result, {})

    def test_invalid_json(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            f.flush()
            h = MCPHandlers()
            result = h._load_mcp_servers(Path(f.name))
            self.assertEqual(result, {})

    def test_no_mcp_servers_key(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"other": "data"}, f)
            f.flush()
            h = MCPHandlers()
            result = h._load_mcp_servers(Path(f.name))
            self.assertEqual(result, {})


class TestMCPWriteMcpServers(unittest.TestCase):
    def test_write_and_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            h = MCPHandlers()
            servers = {"srv1": {"type": "stdio", "command": "x"}}
            h._write_mcp_servers(path, servers)
            result = h._load_mcp_servers(path)
            self.assertEqual(result, servers)

    def test_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            h = MCPHandlers()
            h._write_mcp_servers(path, {"a": {}})
            self.assertFalse((path.with_name(f"{path.name}.tmp")).exists())


class TestMCPUpsertMcpServer(unittest.TestCase):
    def test_insert_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            h = MCPHandlers()
            h._upsert_mcp_server(path, "srv1", {"type": "stdio"})
            result = h._load_mcp_servers(path)
            self.assertIn("srv1", result)

    def test_update_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            h = MCPHandlers()
            h._upsert_mcp_server(path, "srv1", {"type": "stdio"})
            h._upsert_mcp_server(path, "srv1", {"type": "sse", "url": "http://x"})
            result = h._load_mcp_servers(path)
            self.assertEqual(result["srv1"]["type"], "sse")


class TestMCPResolveTargetName(unittest.TestCase):
    def test_exact_match(self) -> None:
        h = MCPHandlers()
        servers = {"my-server": {}, "other": {}}
        result = h._resolve_target_name(servers, action="test", name="my-server")
        self.assertEqual(result, "my-server")

    def test_case_insensitive_match(self) -> None:
        h = MCPHandlers()
        servers = {"My-Server": {}, "other": {}}
        result = h._resolve_target_name(servers, action="test", name="my-server")
        self.assertEqual(result, "My-Server")

    def test_not_found(self) -> None:
        h = MCPHandlers()
        servers = {"my-server": {}}
        result = h._resolve_target_name(servers, action="test", name="nonexistent")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

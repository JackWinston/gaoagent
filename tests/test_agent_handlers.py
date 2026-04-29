from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import uvicorn
    from gaoagent.agent.agent_handlers import AgentHandlers
    HAS_UVICORN = True
except ImportError:
    HAS_UVICORN = False


@unittest.skipUnless(HAS_UVICORN, "uvicorn not installed")
class TestAgentLoadAgents(unittest.TestCase):
    def test_valid_config(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"agents": {"agent1": {"url": "http://x"}}}, f)
            f.flush()
            h = AgentHandlers()
            result = h._load_agents(Path(f.name))
            self.assertIn("agent1", result)

    def test_missing_file(self) -> None:
        h = AgentHandlers()
        result = h._load_agents(Path("/nonexistent/file.json"))
        self.assertEqual(result, {})

    def test_invalid_json(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            f.flush()
            h = AgentHandlers()
            result = h._load_agents(Path(f.name))
            self.assertEqual(result, {})


@unittest.skipUnless(HAS_UVICORN, "uvicorn not installed")
class TestAgentWriteAgents(unittest.TestCase):
    def test_write_and_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            h = AgentHandlers()
            agents = {"agent1": {"url": "http://localhost"}}
            h._write_agents(path, agents)
            result = h._load_agents(path)
            self.assertEqual(result, agents)

    def test_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            h = AgentHandlers()
            h._write_agents(path, {"a": {"url": "http://x"}})
            self.assertFalse((path.with_name(f"{path.name}.tmp")).exists())


if __name__ == "__main__":
    unittest.main()

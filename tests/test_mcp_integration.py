import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestMcpIntegration(unittest.TestCase):
    def test_build_mcp_tools_cache_payload(self):
        from gaoagent.mcp.MCPClientCompat import build_mcp_tools_cache_payload

        mcp_servers = {
            "bing-search": {
                "disabled": False,
                "timeout": 60,
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "bing-cn-mcp"],
            }
        }

        def _fake_list(_name, _body):
            return [
                {
                    "name": "search",
                    "description": "Bing search",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ]

        payload = build_mcp_tools_cache_payload(
            mcp_servers,
            connect_and_list_tools=_fake_list,
            generated_at="test",
        )
        self.assertEqual(payload.get("version"), 1)
        exported_map = payload.get("exported_map")
        self.assertIsInstance(exported_map, dict)
        self.assertEqual(len(exported_map), 1)
        exported_name = next(iter(exported_map.keys()))
        self.assertTrue(exported_name.startswith("mcp__bing-search__"))
        self.assertEqual(exported_map[exported_name]["server"], "bing-search")
        self.assertEqual(exported_map[exported_name]["tool"], "search")

    def test_utils_load_mcp_servers_raw_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "demo_project"
            cfg_dir = project_root / ".gaoagent"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "gao_client_mcp_setting.json").write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "fetch": {
                                "disabled": False,
                                "timeout": 60,
                                "type": "stdio",
                                "command": "uvx",
                                "args": ["mcp-server-fetch"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (cfg_dir / "gao_client_mcp_tools_cache.json").write_text(
                json.dumps({"exported_map": {"mcp__fetch__fetch": {"server": "fetch", "tool": "fetch"}}}),
                encoding="utf-8",
            )

            from gaoagent.core.runner.Utils import load_mcp, load_mcp_servers_raw, load_mcp_tools_cache

            old_cwd = Path.cwd()
            os.chdir(project_root)
            try:
                raw = load_mcp_servers_raw()
                self.assertIn("fetch", raw)
                self.assertEqual(raw["fetch"]["command"], "uvx")

                public = load_mcp()
                self.assertTrue(public.get("available"))
                self.assertEqual(public["servers"][0]["name"], "fetch")

                cache = load_mcp_tools_cache()
                self.assertIsInstance(cache, dict)
                self.assertIn("exported_map", cache)
            finally:
                os.chdir(old_cwd)

    def test_build_function_specs_with_mcp(self):
        from gaoagent.core.runner.FunctionCallProtocol import build_function_specs

        exported_name = "mcp__bing-search__search"
        specs = build_function_specs(
            ["read_file", exported_name],
            mcp_exported_map={
                exported_name: {
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                }
            },
        )
        mcp_spec = [x for x in specs if x["function"]["name"] == exported_name][0]
        self.assertEqual(mcp_spec["function"]["description"], "Search")
        self.assertIn("query", mcp_spec["function"]["parameters"]["properties"])

    def test_react_runner_filter_exported_map_by_enabled_servers(self):
        from gaoagent.core.runner.ReActRunner import ReActRunner

        exported_map = {
            "mcp__project__a": {"server": "project", "tool": "a"},
            "mcp__global__b": {"server": "global", "tool": "b"},
            "mcp__bad__c": {"server": "", "tool": "c"},
        }
        enabled = {"project": {"disabled": False}}
        filtered = ReActRunner._filter_exported_map_for_servers(exported_map, enabled)
        self.assertEqual(set(filtered.keys()), {"mcp__project__a"})

    def test_react_runner_ignores_all_disabled_mcp_servers(self):
        from gaoagent.core.runner.BaseRunner import RequestBaseInfo, StepResult
        from gaoagent.core.runner.ReActRunner import ReActRunner

        runner = ReActRunner()
        runner.request_base_info = RequestBaseInfo(baseurl="http://mock", api_key="mock", modules="mock")

        with (
            patch(
                "gaoagent.core.runner.ReActRunner.load_mcp_servers_raw",
                return_value={
                    "fetch": {
                        "disabled": True,
                        "timeout": 60,
                        "type": "stdio",
                        "command": "uvx",
                        "args": ["mcp-server-fetch"],
                    }
                },
            ),
            patch("gaoagent.core.runner.ReActRunner.load_mcp_tools_cache", return_value=None),
            patch.object(
                ReActRunner,
                "_callLLM",
                return_value=StepResult(decision="final", content="ok"),
            ),
        ):
            result = runner.run("hello")
        self.assertTrue(result.success)
        self.assertEqual(result.final_result, "ok")

    def test_mcp_client_from_config_supports_sse_and_streamable_http(self):
        from gaoagent.mcp.MCPClientCompat import MCPStdioClientSync

        sse_client = MCPStdioClientSync.from_config(
            server_name="sse_server",
            config={
                "type": "sse",
                "url": "https://example.com/sse",
                "timeout": 30,
                "disabled": False,
                "headers": {"Authorization": "Bearer x"},
            },
        )
        self.assertEqual(sse_client.transport_type, "sse")
        self.assertEqual(sse_client.url, "https://example.com/sse")

        http_client = MCPStdioClientSync.from_config(
            server_name="http_server",
            config={
                "type": "streamable_http",
                "url": "https://example.com/mcp",
                "timeout": 30,
                "disabled": False,
            },
        )
        self.assertEqual(http_client.transport_type, "streamable_http")
        self.assertEqual(http_client.url, "https://example.com/mcp")

    def test_mcp_client_from_config_supports_stdio_env(self):
        from gaoagent.mcp.MCPClientCompat import MCPStdioClientSync

        stdio_client = MCPStdioClientSync.from_config(
            server_name="stdio_server",
            config={
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "some-mcp"],
                "env": {"HTTP_PROXY": "http://127.0.0.1:7890"},
                "timeout": 30,
                "disabled": False,
            },
        )
        self.assertEqual(stdio_client.transport_type, "stdio")
        self.assertEqual(stdio_client.command, "npx")
        self.assertEqual(stdio_client.args, ["-y", "some-mcp"])
        self.assertEqual(stdio_client.env, {"HTTP_PROXY": "http://127.0.0.1:7890"})

    def test_core_config_validate_mcp_config_supports_http_types(self):
        from gaoagent.core.CoreConfigDefault import CoreConfigDefault

        validator = CoreConfigDefault()
        validator._validate_mcp_config(
            {
                "demo-sse": {
                    "disabled": False,
                    "timeout": 30,
                    "type": "sse",
                    "url": "https://example.com/sse",
                    "headers": {"Authorization": "Bearer token"},
                }
            }
        )
        validator._validate_mcp_config(
            {
                "demo-http": {
                    "disabled": False,
                    "timeout": 30,
                    "type": "streamable_http",
                    "url": "https://example.com/mcp",
                }
            }
        )
        validator._validate_mcp_config(
            {
                "demo-stdio": {
                    "disabled": False,
                    "timeout": 30,
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "bing-cn-mcp"],
                    "env": {"HTTP_PROXY": "http://127.0.0.1:7890"},
                }
            }
        )
        with self.assertRaises(ValueError):
            validator._validate_mcp_config(
                {
                    "bad-http": {
                        "disabled": False,
                        "timeout": 30,
                        "type": "streamable_http",
                    }
                }
            )
        with self.assertRaises(ValueError):
            validator._validate_mcp_config(
                {
                    "bad-stdio-env": {
                        "disabled": False,
                        "timeout": 30,
                        "type": "stdio",
                        "command": "npx",
                        "args": ["-y", "bing-cn-mcp"],
                        "env": {"HTTP_PROXY": 7890},
                    }
                }
            )

if __name__ == "__main__":
    unittest.main()

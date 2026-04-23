from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Any, Callable

import click

from gaoagent.core.runner.Utils import safe_json_dumps, try_project_root_dir


@dataclass(frozen=True)
class ToolCall:
    """ToolCall 类。
    
    职责:
    - 封装该模块内相关的业务能力与状态。
    - 提供 ToolCall 语义下的方法集合，供上层流程协调调用。
    
    继承关系:
    - 基类: 无
    """
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tool_call_id: str | None = None


ToolHandler = Callable[[Any, dict[str, Any]], Any]


class ToolRegistry:
    """ToolRegistry 类。
    
    职责:
    - 封装该模块内相关的业务能力与状态。
    - 提供 ToolRegistry 语义下的方法集合，供上层流程协调调用。
    
    继承关系:
    - 基类: 无
    """
    def __init__(self) -> None:
        """__init__ 方法。
        
        用途:
        - 执行当前步骤的核心逻辑，并与调用链中的上下文保持一致。
        
        参数:
        - 无: 该方法不需要额外业务参数。
        
        返回:
        - None: 构造函数仅完成实例初始化，不返回业务结果。
        """
        self._tools: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        """register 方法。
        
        用途:
        - 执行当前步骤的核心逻辑，并与调用链中的上下文保持一致。
        
        参数:
        - name: 输入参数，用于控制该方法的处理行为。
        - handler: 输入参数，用于控制该方法的处理行为。
        
        返回:
        - Any: 返回当前步骤产出的结果；具体结构由调用方约定。
        """
        if not name or not isinstance(name, str):
            raise ValueError("tool name must be non-empty str")
        self._tools[name] = handler

    def call(self, ctx: Any, call: ToolCall) -> str:
        """call 方法。
        
        用途:
        - 执行当前步骤的核心逻辑，并与调用链中的上下文保持一致。
        
        参数:
        - ctx: 输入参数，用于控制该方法的处理行为。
        - call: 输入参数，用于控制该方法的处理行为。
        
        返回:
        - Any: 返回当前步骤产出的结果；具体结构由调用方约定。
        """
        handler = self._tools.get(call.name)
        if handler is None:
            raise KeyError(f"tool not found: {call.name}")
        raw = handler(ctx, call.arguments)
        try:
            setattr(ctx, "last_observation_raw", raw)
        except Exception:
            pass
        content = raw if isinstance(raw, str) else safe_json_dumps(raw)
        return content

    def list_names(self) -> list[str]:
        """list_names 方法。
        
        用途:
        - 执行当前步骤的核心逻辑，并与调用链中的上下文保持一致。
        
        参数:
        - 无: 该方法不需要额外业务参数。
        
        返回:
        - Any: 返回当前步骤产出的结果；具体结构由调用方约定。
        """
        return sorted(self._tools.keys())


def default_tool_registry() -> ToolRegistry:
    """default_tool_registry 函数。
    
    用途:
    - 执行当前步骤的核心逻辑，并与调用链中的上下文保持一致。
    
    参数:
    - 无: 该方法不需要额外业务参数。
    
    返回:
    - Any: 返回当前步骤产出的结果；具体结构由调用方约定。
    """
    tools = ToolRegistry()

    def _list_dir(_ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
        """_list_dir 函数。
        
        用途:
        - 执行当前步骤的核心逻辑，并与调用链中的上下文保持一致。
        
        参数:
        - _ctx: 输入参数，用于控制该函数的处理行为。
        - args: 输入参数，用于控制该函数的处理行为。
        
        返回:
        - Any: 返回当前步骤产出的结果；具体结构由调用方约定。
        """
        path = args.get("path", ".")
        if not isinstance(path, str) or not path.strip():
            return {
                "ok": False,
                "error": {
                    "type": "ValueError",
                    "message": "path must be non-empty str",
                },
            }
        try:
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if not p.exists():
                return {
                    "success": False,
                    "error": {"type": "FileNotFoundError", "message": "path not found"},
                    "path": str(p),
                }
            if not p.is_dir():
                return {
                    "success": False,
                    "error": {
                        "type": "NotADirectoryError",
                        "message": "path is not a directory",
                    },
                    "path": str(p),
                }
            items: list[dict[str, Any]] = []
            for child in p.iterdir():
                try:
                    st = child.stat()
                    size = int(st.st_size)
                except Exception:
                    size = None
                items.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "is_dir": child.is_dir(),
                        "size": size,
                    }
                )
            items.sort(
                key=lambda x: (
                    not bool(x.get("is_dir")),
                    str(x.get("name") or "").lower(),
                )
            )
            return {"success": True, "path": str(p), "items": items}
        except Exception as e:
            return {
                "success": False,
                "error": {"type": type(e).__name__, "message": str(e)},
                "path": str(path),
            }

    def _read_file(_ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
        """_read_file 函数。
        
        用途:
        - 执行当前步骤的核心逻辑，并与调用链中的上下文保持一致。
        
        参数:
        - _ctx: 输入参数，用于控制该函数的处理行为。
        - args: 输入参数，用于控制该函数的处理行为。
        
        返回:
        - Any: 返回当前步骤产出的结果；具体结构由调用方约定。
        """
        path = args.get("path")
        encoding = args.get("encoding") or "utf-8"
        if not isinstance(path, str) or not path.strip():
            return {
                "success": False,
                "error": {
                    "type": "ValueError",
                    "message": "path must be non-empty str",
                },
            }
        if not isinstance(encoding, str) or not encoding.strip():
            return {
                "success": False,
                "error": {
                    "type": "ValueError",
                    "message": "encoding must be non-empty str",
                },
            }
        try:
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            content = p.read_text(encoding=encoding)
            return {
                "success": True,
                "path": str(p),
                "encoding": encoding,
                "content": content,
            }
        except Exception as e:
            return {
                "success": False,
                "error": {"type": type(e).__name__, "message": str(e)},
                "path": str(path),
            }

    def _ask_user(_ctx: Any, args: dict[str, Any]) -> str:
        """_ask_user 函数。
        
        用途:
        - 执行当前步骤的核心逻辑，并与调用链中的上下文保持一致。
        
        参数:
        - _ctx: 输入参数，用于控制该函数的处理行为。
        - args: 输入参数，用于控制该函数的处理行为。
        
        返回:
        - Any: 返回当前步骤产出的结果；具体结构由调用方约定。
        """
        prompt = args.get("prompt")
        default = args.get("default", None)
        choices = args.get("choices", None)
        if not isinstance(prompt, str) or not prompt.strip():
            return safe_json_dumps(
                {
                    "success": False,
                    "error": {
                        "type": "ValueError",
                        "message": "prompt must be non-empty str",
                    },
                }
            )
        if default is not None and not isinstance(default, str):
            default = str(default)
        if choices is not None and not isinstance(choices, list):
            return safe_json_dumps(
                {
                    "success": False,
                    "error": {
                        "type": "ValueError",
                        "message": "choices must be list[str]",
                    },
                }
            )
        if isinstance(choices, list) and any(not isinstance(x, str) for x in choices):
            return safe_json_dumps(
                {
                    "success": False,
                    "error": {
                        "type": "ValueError",
                        "message": "choices must be list[str]",
                    },
                }
            )
        try:
            typ = (
                click.Choice(choices) if isinstance(choices, list) and choices else str
            )
            click.echo(prompt.strip())
            answer = click.prompt("", default=default, type=typ, prompt_suffix="")
            return str(answer)
        except Exception as e:
            return safe_json_dumps(
                {
                    "success": False,
                    "error": {"type": type(e).__name__, "message": str(e)},
                    "prompt": str(prompt),
                }
            )

    def _write_file(_ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
        """_write_file 函数。
        
        用途:
        - 执行当前步骤的核心逻辑，并与调用链中的上下文保持一致。
        
        参数:
        - _ctx: 输入参数，用于控制该函数的处理行为。
        - args: 输入参数，用于控制该函数的处理行为。
        
        返回:
        - Any: 返回当前步骤产出的结果；具体结构由调用方约定。
        """
        path = args.get("path")
        content = args.get("content", "")
        encoding = args.get("encoding") or "utf-8"
        mkdirs = args.get("mkdirs", True)
        append = args.get("append", False)
        if not isinstance(path, str) or not path.strip():
            return {
                "success": False,
                "error": {
                    "type": "ValueError",
                    "message": "path must be non-empty str",
                },
            }
        if not isinstance(encoding, str) or not encoding.strip():
            return {
                "success": False,
                "error": {
                    "type": "ValueError",
                    "message": "encoding must be non-empty str",
                },
            }
        if not isinstance(mkdirs, bool):
            return {
                "success": False,
                "error": {"type": "ValueError", "message": "mkdirs must be bool"},
            }
        if not isinstance(append, bool):
            return {
                "success": False,
                "error": {"type": "ValueError", "message": "append must be bool"},
            }
        try:
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if mkdirs:
                p.parent.mkdir(parents=True, exist_ok=True)
            text = content if isinstance(content, str) else str(content)
            mode = "a" if append else "w"
            with p.open(mode, encoding=encoding) as f:
                n = f.write(text)
            return {
                "success": True,
                "path": str(p),
                "encoding": encoding,
                "written_chars": n,
                "append": append,
            }
        except Exception as e:
            return {
                "success": False,
                "error": {"type": type(e).__name__, "message": str(e)},
                "path": str(path),
            }

    def _run_command(_ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
        """_run_command 函数。
        
        用途:
        - 执行当前步骤的核心逻辑，并与调用链中的上下文保持一致。
        
        参数:
        - _ctx: 输入参数，用于控制该函数的处理行为。
        - args: 输入参数，用于控制该函数的处理行为。
        
        返回:
        - Any: 返回当前步骤产出的结果；具体结构由调用方约定。
        """
        workdir = args.get("workdir")
        command = args.get("command")
        if not isinstance(workdir, str) or not workdir.strip():
            return {
                "success": False,
                "error": {
                    "type": "ValueError",
                    "message": "workdir must be non-empty str",
                },
            }
        if not isinstance(command, str) or not command.strip():
            return {
                "success": False,
                "error": {
                    "type": "ValueError",
                    "message": "command must be non-empty str",
                },
            }

        try:
            cwd = Path.cwd().resolve()
            target = Path(workdir).expanduser()
            if not target.is_absolute():
                target = (cwd / target).resolve()
            else:
                target = target.resolve()

            if not target.exists():
                return {
                    "success": False,
                    "error": {
                        "type": "FileNotFoundError",
                        "message": "workdir not found",
                    },
                    "workdir": str(target),
                }
            if not target.is_dir():
                return {
                    "success": False,
                    "error": {
                        "type": "NotADirectoryError",
                        "message": "workdir is not a directory",
                    },
                    "workdir": str(target),
                }
            if target != cwd and cwd not in target.parents:
                return {
                    "success": False,
                    "error": {
                        "type": "PermissionError",
                        "message": "workdir must be current directory or its subdirectory",
                    },
                    "workdir": str(target),
                    "cwd": str(cwd),
                }

            completed = subprocess.run(
                command,
                cwd=str(target),
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            return {
                "success": completed.returncode == 0,
                "workdir": str(target),
                "command": command,
                "exit_code": int(completed.returncode),
                "stdout": stdout,
                "stderr": stderr,
                "response": f"{stdout}{stderr}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": {"type": type(e).__name__, "message": str(e)},
                "workdir": str(workdir),
                "command": str(command),
            }

    def _rag_search(_ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
        """_rag_search 函数。
        
        用途:
        - 执行当前步骤的核心逻辑，并与调用链中的上下文保持一致。
        
        参数:
        - _ctx: 输入参数，用于控制该函数的处理行为。
        - args: 输入参数，用于控制该函数的处理行为。
        
        返回:
        - Any: 返回当前步骤产出的结果；具体结构由调用方约定。
        """
        kb_name = args.get("kb_name")
        query = args.get("query")
        top_k = args.get("top_k", 5)
        
        if not isinstance(kb_name, str) or not kb_name.strip():
            return {"success": False, "error": "kb_name 不能为空"}
        if not isinstance(query, str) or not query.strip():
            return {"success": False, "error": "query 不能为空"}
            
        try:
            from gaoagent.rag.RagChromaRetriever import RagChromaRetriever
            retriever = RagChromaRetriever(kb_name=kb_name)
            return retriever.search(query=query, top_k=int(top_k))
        except Exception as e:
            return {"success": False, "error": f"检索异常：{str(e)}"}

    def _search_workspace(_ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
        """在当前项目范围内执行全文检索（基于 ripgrep）。"""
        query = args.get("query")
        scope_path = args.get("scope_path")
        file_glob = args.get("file_glob")
        max_results = args.get("max_results", 50)
        case_sensitive = args.get("case_sensitive", False)
        literal = args.get("literal", False)

        if not isinstance(query, str) or not query.strip():
            return {
                "success": False,
                "error": {"type": "ValueError", "message": "query must be non-empty str"},
            }
        if not isinstance(max_results, int):
            return {
                "success": False,
                "error": {"type": "ValueError", "message": "max_results must be int"},
            }
        if max_results <= 0:
            return {
                "success": False,
                "error": {"type": "ValueError", "message": "max_results must be > 0"},
            }
        if max_results > 500:
            max_results = 500
        if not isinstance(case_sensitive, bool):
            return {
                "success": False,
                "error": {"type": "ValueError", "message": "case_sensitive must be bool"},
            }
        if not isinstance(literal, bool):
            return {
                "success": False,
                "error": {"type": "ValueError", "message": "literal must be bool"},
            }

        globs: list[str] = []
        if isinstance(file_glob, str) and file_glob.strip():
            globs.append(file_glob.strip())
        elif isinstance(file_glob, list):
            for item in file_glob:
                if isinstance(item, str) and item.strip():
                    globs.append(item.strip())
                else:
                    return {
                        "success": False,
                        "error": {
                            "type": "ValueError",
                            "message": "file_glob list items must be non-empty str",
                        },
                    }
        elif file_glob is not None:
            return {
                "success": False,
                "error": {"type": "ValueError", "message": "file_glob must be str or list[str]"},
            }

        project_root = try_project_root_dir() or Path.cwd().resolve()
        if not project_root.exists() or not project_root.is_dir():
            return {
                "success": False,
                "error": {"type": "NotADirectoryError", "message": "project root is invalid"},
                "project_root": str(project_root),
            }

        search_root = project_root
        search_target = "."
        scope_type = "project"
        normalized_scope_path = ""
        if scope_path is not None:
            if not isinstance(scope_path, str):
                return {
                    "success": False,
                    "error": {"type": "ValueError", "message": "scope_path must be str"},
                }
            normalized_scope_path = scope_path.strip().replace("\\", "/")
            if normalized_scope_path:
                candidate = Path(normalized_scope_path)
                if not candidate.is_absolute():
                    return {
                        "success": False,
                        "error": {
                            "type": "ValueError",
                            "message": "scope_path must be an absolute path",
                        },
                        "project_root": str(project_root),
                    }
                search_root = candidate.resolve()
                if search_root != project_root and project_root not in search_root.parents:
                    return {
                        "success": False,
                        "error": {
                            "type": "PermissionError",
                            "message": "scope_path must stay inside project root",
                        },
                        "project_root": str(project_root),
                    }
                if not search_root.exists():
                    return {
                        "success": False,
                        "error": {"type": "FileNotFoundError", "message": "scope_path not found"},
                        "project_root": str(project_root),
                        "scope_path": normalized_scope_path,
                    }
                if search_root.is_dir():
                    search_target = "."
                    scope_type = "directory"
                elif search_root.is_file():
                    search_target = search_root.name
                    search_root = search_root.parent
                    scope_type = "file"
                else:
                    return {
                        "success": False,
                        "error": {
                            "type": "ValueError",
                            "message": "scope_path must point to a file or directory",
                        },
                        "project_root": str(project_root),
                        "scope_path": normalized_scope_path,
                    }

        cmd: list[str] = [
            "rg",
            "--json",
            "--line-number",
            "--column",
            "--with-filename",
            "--no-heading",
            "--glob",
            "!.git/*",
        ]
        if case_sensitive:
            cmd.append("--case-sensitive")
        else:
            cmd.append("--smart-case")
        if literal:
            cmd.append("--fixed-strings")
        for pattern in globs:
            cmd.extend(["--glob", pattern])
        cmd.extend([query.strip(), search_target])

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(search_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            return {
                "success": False,
                "error": {"type": "FileNotFoundError", "message": "ripgrep (rg) not found"},
                "project_root": str(project_root),
            }
        except Exception as e:
            return {
                "success": False,
                "error": {"type": type(e).__name__, "message": str(e)},
                "project_root": str(project_root),
            }

        results: list[dict[str, Any]] = []
        stopped_early = False
        try:
            assert process.stdout is not None
            for line in process.stdout:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except Exception:
                    continue
                if event.get("type") != "match":
                    continue
                data = event.get("data") if isinstance(event.get("data"), dict) else {}
                line_no = data.get("line_number")
                line_text = (
                    data.get("lines", {}).get("text")
                    if isinstance(data.get("lines"), dict)
                    else ""
                )
                submatches = data.get("submatches") if isinstance(data.get("submatches"), list) else []
                path_text = (
                    data.get("path", {}).get("text")
                    if isinstance(data.get("path"), dict)
                    else None
                )
                if not isinstance(path_text, str) or not path_text:
                    continue

                abs_path = (search_root / path_text).resolve()
                if abs_path != project_root and project_root not in abs_path.parents:
                    continue

                if submatches:
                    for sm in submatches:
                        if not isinstance(sm, dict):
                            continue
                        results.append(
                            {
                                "path": str(abs_path),
                                "line": int(line_no) if isinstance(line_no, int) else None,
                                "start": sm.get("start"),
                                "end": sm.get("end"),
                                "line_text": line_text,
                            }
                        )
                        if len(results) >= max_results:
                            stopped_early = True
                            break
                else:
                    results.append(
                        {
                            "path": str(abs_path),
                            "line": int(line_no) if isinstance(line_no, int) else None,
                            "start": None,
                            "end": None,
                            "line_text": line_text,
                        }
                    )
                if len(results) >= max_results:
                    stopped_early = True
                    break
        finally:
            if stopped_early:
                process.terminate()
            try:
                stdout_text, stderr_text = process.communicate(timeout=1)
            except Exception:
                process.kill()
                stdout_text, stderr_text = process.communicate()

        exit_code = int(process.returncode) if process.returncode is not None else -1
        # rg 约定：0=有匹配，1=无匹配，2=执行错误
        if not stopped_early and exit_code not in (0, 1):
            return {
                "success": False,
                "error": {
                    "type": "RuntimeError",
                    "message": (stderr_text or stdout_text or "rg failed").strip(),
                },
                "project_root": str(project_root),
                "exit_code": exit_code,
            }

        return {
            "success": True,
            "project_root": str(project_root),
            "search_root": str(search_root),
            "scope_path": normalized_scope_path,
            "scope_type": scope_type,
            "search_target": search_target,
            "query": query.strip(),
            "used_gitignore": True,
            "max_results": max_results,
            "count": len(results),
            "results": results,
        }

    tools.register("list_dir", _list_dir)
    tools.register("read_file", _read_file)
    tools.register("ask_user", _ask_user)
    tools.register("write_file", _write_file)
    tools.register("run_command", _run_command)
    tools.register("search_workspace", _search_workspace)
    tools.register("rag_search", _rag_search)
    return tools

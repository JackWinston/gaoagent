from __future__ import annotations

import os
import threading
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gaoagent.core.runner.utils import project_config_dir, safe_json_dumps


class RunLogger:
    """RunLogger 类。
    
    职责:
    - 记录 ReAct 模式的运行日志。
    
    """
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self._lock = threading.Lock()

    def log_event(self, event_type: str, payload: Any, *, step: int | None = None) -> None:
        """log_event 方法。
        
        用途:
        - 记录 ReAct 模式的运行事件。
        
        参数:
        - event_type: 输入参数，用于指定事件类型，必须是非空字符串；空值会被直接拒绝。
        - payload: 输入参数，用于指定事件数据。
        - step: 输入参数，用于指定当前事件的步骤号。
        
        """
        record: dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event_type,
        }
        if step is not None:
            record["step"] = step
        record["data"] = _to_jsonable(payload)
        line = safe_json_dumps(record)
        with self._lock:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n\n")


_CURRENT_RUN_LOGGER: ContextVar[RunLogger | None] = ContextVar("gaoagent_run_logger", default=None)


def get_current_run_logger() -> RunLogger | None:
    """get_current_run_logger 函数。
    
    用途:
    - 获取当前运行日志记录器。
    
    返回:
    - RunLogger | None: 返回当前运行日志记录器；如果未设置，则返回   None。
    """
    return _CURRENT_RUN_LOGGER.get()


def set_current_run_logger(logger: RunLogger | None):
    """set_current_run_logger 函数。
    
    用途:
    - 设置当前运行日志记录器。
    
    参数:
    - logger: 输入参数，用于指定要设置的运行日志记录器。
    

    """
    return _CURRENT_RUN_LOGGER.set(logger)


def reset_current_run_logger(token) -> None:
    """reset_current_run_logger 函数。
    
    用途:
    - 重置当前运行日志记录器。
    
    参数:
    - token: 输入参数，用于控制该函数的处理行为。
    
    """
    _CURRENT_RUN_LOGGER.reset(token)


def create_run_logger() -> RunLogger:
    """create_run_logger 函数。
    
    用途:
    - 创建一个新的 RunLogger 实例，用于记录 ReAct 模式的运行日志。
    
    参数:
    - 无: 该方法不需要额外业务参数。
    
    返回:
    - RunLogger: 返回一个新的 RunLogger 实例。
    """
    cfg_dir = project_config_dir()
    if cfg_dir is None:
        raise RuntimeError("当前目录或其父目录未初始化 GaoAgent 项目。请先运行 `gaoagent init`。")
    logs_dir = cfg_dir / "logs"
    ts = datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
    if os.name == "nt":
        ts = ts.replace(":", "-")
    file_path = logs_dir / f"{ts}.log"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)
    return RunLogger(file_path)


def _to_jsonable(value: Any) -> Any:
    """_to_jsonable 函数。
    
    用途:
    - 将任意 Python 对象转换为 JSON 可序列化的格式。
    
    参数:
    - value: 输入参数，用于指定待转换的 Python 对象。
    
    返回:
    - Any: 返回转换后的 JSON 可序列化对象。
    """
    if is_dataclass(value):
        try:
            return {k: _to_jsonable(v) for k, v in asdict(value).items()}
        except Exception as e:
            from gaoagent.core.runner.console import Console
            Console.debug(f"dataclass 序列化失败，回退 repr：{e}")
            return repr(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            out[key] = _to_jsonable(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


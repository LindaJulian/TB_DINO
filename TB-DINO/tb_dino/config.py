from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is invalid."""


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"无法解析配置覆盖值: {value}") from exc


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = config
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            current = {}
            cursor[part] = current
        if not isinstance(current, dict):
            raise ConfigError(f"配置项 {part} 不是字典，无法设置 {key}")
        cursor = current
    cursor[parts[-1]] = value


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Load YAML with optional ``_base_`` inheritance and dotted overrides."""

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是字典: {path}")

    base_ref = data.pop("_base_", None)
    if base_ref:
        base_path = (path.parent / str(base_ref)).resolve()
        config = _deep_merge(load_config(base_path), data)
    else:
        config = data

    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"覆盖项必须采用 key=value 格式: {item}")
        key, raw_value = item.split("=", 1)
        _set_dotted(config, key.strip(), _parse_scalar(raw_value))

    config["_config_path"] = str(path)
    return config


def get_required(config: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise ConfigError(f"缺少必需配置项: {dotted_key}")
        cursor = cursor[part]
    return cursor


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path.resolve()


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(serializable, file, allow_unicode=True, sort_keys=False)


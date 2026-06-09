"""配置结构和 JSON 加载。"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class ServerConfig:
    host: str
    port: int
    log_level: str


@dataclass
class DatasetConfig:
    base_path: str
    dimension: int
    max_vectors: int | None


@dataclass
class HNSWConfig:
    default_k: int
    candidate_k: int
    hnsw_index_path: str
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int


@dataclass
class PipelineConfig:
    local_search_workers: int
    rerank_workers: int


@dataclass
class SearchConfig:
    hnsw: HNSWConfig
    pipeline: PipelineConfig
    offload_qps_threshold: float
    force_faas: bool


@dataclass
class ScalingConfig:
    prewarm_check_seconds: float
    load_index_timeout_seconds: float
    enable_prewarm: bool
    local_candidate_ms: float
    remote_candidate_ms: float
    function_concurrency: int
    max_warm_functions: int
    function_memory_mb: int
    cost_per_gb_second: float


@dataclass
class FaaSConfig:
    provider: str
    invoke_timeout_seconds: float
    warmup_timeout_seconds: float
    endpoints: dict[str, str]


@dataclass
class AppConfig:
    server: ServerConfig
    dataset: DatasetConfig
    search: SearchConfig
    scaling: ScalingConfig
    faas: FaaSConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = _as_dict(json.loads(config_path.read_text(encoding="utf-8")), str(config_path))
    section_types = {
        "server": ServerConfig,
        "dataset": DatasetConfig,
        "search": SearchConfig,
        "scaling": ScalingConfig,
        "faas": FaaSConfig,
    }
    _reject_unknown(raw, set(section_types), str(config_path))

    values = {}
    for name, cls in section_types.items():
        if name not in raw:
            raise ValueError(f"{config_path}: missing required config section {name!r}")
        section = _as_dict(raw[name], f"{config_path}.{name}")
        if name == "search":
            hnsw_data = _as_dict(section.get("hnsw"), f"{config_path}.search.hnsw")
            section["hnsw"] = _build_config(HNSWConfig, hnsw_data, "search.hnsw")
            pipeline_data = _as_dict(section.get("pipeline"), f"{config_path}.search.pipeline")
            section["pipeline"] = _build_config(PipelineConfig, pipeline_data, "search.pipeline")
        values[name] = _build_config(cls, section, name)
    return AppConfig(**values)


def project_path(config_path: str | Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return Path(config_path).resolve().parent.parent / path


def _build_config(cls: type, data: dict, section: str):
    _reject_unknown(data, {field.name for field in fields(cls)}, section)
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError(f"{section}: {exc}") from exc


def _as_dict(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _reject_unknown(data: dict, allowed: set[str], name: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{name}: unknown config key(s): {', '.join(unknown)}")

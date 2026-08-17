"""FaasANN 服务器主入口。

这个文件负责把各模块连接起来：VM 加载 raw vectors 和本地 HNSW，云函数路径只做 PQ 候选召回，
最终 exact rerank 始终在 VM 侧完成。
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI

from faas.aliyun_fc_provider import AliyunHTTPProvider
from faas.local_provider import LocalFaaSProvider
from scaling.metrics import RuntimeMetrics
from scaling.planner import OffloadPlanner
from scaling.prewarm import WarmupManager
from search.hnsw import HNSWIndex
from search.service import SearchService
from server.middleware import add_process_time_header
from server.routes import create_router
from utils.config import AppConfig, load_config, project_path
from utils.logging import configure_logging
from vectors.vector_store import VectorStore

logger = logging.getLogger(__name__)

CONFIG_PATH_ENV = "FAASANN_SERVER_CONFIG"
SERVER_WORKERS_ENV = "FAASANN_SERVER_WORKERS"


def load_vector_store(config: AppConfig, config_path: str) -> VectorStore:
    base_path = project_path(config_path, config.dataset.base_path)
    if not Path(base_path).exists():
        raise FileNotFoundError(f"dataset file not found: {base_path}")

    logger.info("loading vectors from %s", base_path)
    return VectorStore.from_fvecs(
        str(base_path),
        dimension=config.dataset.dimension,
        max_vectors=config.dataset.max_vectors,
    )


def create_app(config_path: str, *, server_workers: int = 1) -> FastAPI:
    # 读取 server/local 或 server/aliyun 配置，并按配置设置日志级别。
    config = load_config(config_path)
    configure_logging(config.server.log_level)
    worker_pipeline = replace(
        config.search.pipeline,
        local_search_workers=_workers_per_process(
            config.search.pipeline.local_search_workers,
            server_workers,
        ),
        faas_invoke_workers=_workers_per_process(
            config.search.pipeline.faas_invoke_workers,
            server_workers,
        ),
        rerank_workers=_workers_per_process(
            config.search.pipeline.rerank_workers,
            server_workers,
        ),
    )
    worker_search_config = replace(config.search, pipeline=worker_pipeline)

    # VM 侧始终加载原始向量；无论候选来自本地还是云函数，最终 exact rerank 都要用它。
    vector_store = load_vector_store(config, config_path)

    # 本地 HNSW 索引用于 local 路径；走阿里云函数时，它仍保留给低 QPS 或调试场景使用。
    local_index = HNSWIndex(
        vector_store.vectors,
        index_path=project_path(config_path, config.search.hnsw.hnsw_index_path),
        m=config.search.hnsw.hnsw_m,
        ef_construction=config.search.hnsw.hnsw_ef_construction,
        ef_search=config.search.hnsw.hnsw_ef_search,
    )

    # provider 决定第一阶段候选召回发到哪里：阿里云 HTTP 函数，或本地 HNSW 模拟器。
    if config.faas.provider == "aliyun_http":
        provider = AliyunHTTPProvider(
            endpoints=config.faas.endpoints,
            timeout_seconds=config.faas.invoke_timeout_seconds,
            invoke_workers=worker_pipeline.faas_invoke_workers,
        )
    elif config.faas.provider == "local":
        provider = LocalFaaSProvider(local_index)
    else:
        raise ValueError(f"unsupported faas.provider={config.faas.provider!r}")

    # 组装运行时状态：QPS 统计、offload 决策、预热管理和两阶段搜索服务。
    metrics = RuntimeMetrics()
    planner = OffloadPlanner(worker_search_config, config.scaling, metrics=metrics)
    warmup_manager = WarmupManager(provider=provider, config=config.scaling)
    search_service = SearchService(
        vectors=vector_store,
        local_index=local_index,
        provider=provider,
        warmup_manager=warmup_manager,
        planner=planner,
        metrics=metrics,
        config=worker_search_config,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 服务启动时启动预热后台任务；关闭时停止预热并回收搜索线程池。
        await warmup_manager.start()
        try:
            yield
        finally:
            await warmup_manager.close()
            search_service.close()

    app = FastAPI(title="FaasANN", version="0.1.0", lifespan=lifespan)
    app.middleware("http")(add_process_time_header)
    app.include_router(create_router(search_service, vector_store))
    app.state.config = config
    app.state.search_service = search_service
    app.state.vector_store = vector_store
    logger.info(
        "server initialized: vectors=%d dim=%d index_backend=%s pid=%d server_workers=%d "
        "local_search_workers=%d faas_invoke_workers=%d rerank_workers=%d",
        vector_store.size,
        vector_store.dimension,
        local_index.backend,
        os.getpid(),
        server_workers,
        worker_pipeline.local_search_workers,
        worker_pipeline.faas_invoke_workers if config.faas.provider == "aliyun_http" else 0,
        worker_pipeline.rerank_workers,
    )
    return app


def _workers_per_process(total_workers: int, server_workers: int) -> int:
    return max(1, (total_workers + server_workers - 1) // server_workers)


def create_configured_app() -> FastAPI:
    config_path = os.environ.get(CONFIG_PATH_ENV)
    if not config_path:
        raise RuntimeError(f"{CONFIG_PATH_ENV} is required for multi-worker startup")
    workers = int(os.environ.get(SERVER_WORKERS_ENV, "1"))
    return create_app(config_path, server_workers=workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FaasANN server")
    parser.add_argument("--config", default="configs/server.local.json", help="path to server config")
    parser.add_argument("--workers", type=int, default=None, help="override server.workers")
    parser.add_argument(
        "--worker-healthcheck-timeout",
        type=int,
        default=None,
        help="seconds Uvicorn waits for a worker health response during startup and runtime",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    workers = args.workers if args.workers is not None else config.server.workers
    worker_healthcheck_timeout = (
        args.worker_healthcheck_timeout
        if args.worker_healthcheck_timeout is not None
        else config.server.worker_healthcheck_timeout_seconds
    )
    if workers <= 0:
        raise SystemExit("workers must be positive")
    if worker_healthcheck_timeout <= 0:
        raise SystemExit("worker healthcheck timeout must be positive")

    import uvicorn

    if workers == 1:
        uvicorn.run(
            create_app(args.config, server_workers=1),
            host=config.server.host,
            port=config.server.port,
            log_level=config.server.log_level,
            reload=False,
        )
        return

    os.environ[CONFIG_PATH_ENV] = str(Path(args.config).resolve())
    os.environ[SERVER_WORKERS_ENV] = str(workers)
    uvicorn.run(
        "main:create_configured_app",
        factory=True,
        workers=workers,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        timeout_worker_healthcheck=worker_healthcheck_timeout,
        reload=False,
    )


if __name__ == "__main__":
    main()

"""FaasANN 服务器主入口。

这个文件负责把各模块连接起来：VM 加载 raw vectors 和本地 HNSW，云函数路径只做 PQ 候选召回，
最终 exact rerank 始终在 VM 侧完成。
"""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
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


def create_app(config_path: str) -> FastAPI:
    config = load_config(config_path)
    configure_logging(config.server.log_level)

    vector_store = load_vector_store(config, config_path)
    local_index = HNSWIndex(
        vector_store.vectors,
        index_path=project_path(config_path, config.search.hnsw.hnsw_index_path),
        m=config.search.hnsw.hnsw_m,
        ef_construction=config.search.hnsw.hnsw_ef_construction,
        ef_search=config.search.hnsw.hnsw_ef_search,
    )

    if config.faas.provider == "aliyun_http":
        provider = AliyunHTTPProvider(
            endpoints=config.faas.endpoints,
            timeout_seconds=config.faas.invoke_timeout_seconds,
        )
    elif config.faas.provider == "local":
        provider = LocalFaaSProvider(local_index)
    else:
        raise ValueError(f"unsupported faas.provider={config.faas.provider!r}")

    metrics = RuntimeMetrics()
    planner = OffloadPlanner(config.search, config.scaling)
    warmup_manager = WarmupManager(provider=provider, config=config.scaling)
    search_service = SearchService(
        vectors=vector_store,
        local_index=local_index,
        provider=provider,
        warmup_manager=warmup_manager,
        planner=planner,
        metrics=metrics,
        config=config.search,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
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
        "server initialized: vectors=%d dim=%d index_backend=%s",
        vector_store.size,
        vector_store.dimension,
        local_index.backend,
    )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FaasANN server")
    parser.add_argument("--config", default="configs/server.local.json", help="path to server config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    app = create_app(args.config)

    import uvicorn

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()

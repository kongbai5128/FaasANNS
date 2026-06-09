"""日志初始化工具。

configure_logging 根据配置设置全局 logging 级别和输出格式。
把日志初始化集中起来可以避免各模块各自配置导致输出混乱。
"""

from __future__ import annotations

import logging


def configure_logging(level: str = "info") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

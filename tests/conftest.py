"""测试 import path 配置。

测试运行时把项目 `src` 目录加入 sys.path，使 server/search/vectors/scaling 等源码包可以直接导入。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

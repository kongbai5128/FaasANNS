# 01. ECS 部署常驻服务器

这一阶段的目标是：把当前 FaasANN FastAPI 服务部署到阿里云 ECS，并能访问：

```text
GET  /health
POST /search
```

## 1. 选择 ECS

建议优先选择内存较大的实例。

当前数据规模：

```text
sift_base.fvecs       约 516 MB
full_hnsw.bin         约 752 MB
raw vector matrix     约 512 MB
运行时索引和 Python 开销  可能数 GB
```

建议：

```text
最低：8 GB 内存
更稳：16 GB 或以上
```

系统建议：

```text
Ubuntu 22.04 / 24.04
Python 3.10+
```

## 2. 安全组

如果只从 ECS 本机测试：

```text
不需要开放 8080 公网端口
```

如果要从本地电脑访问：

```text
开放 TCP 8080
来源 IP 建议限制为你的本机公网 IP
```

不要直接对全网开放长期服务，除非加了鉴权、网关或反向代理。

## 3. 上传代码和数据

可以用以下几种方式：

```text
方式 A：git clone 项目代码，然后单独上传 data/
方式 B：scp/rsync 整个 /home/qian/Code/FaasANN
方式 C：代码走 git，数据放 OSS，再下载到 ECS
```

推荐目录：

```text
/home/admin/FaasANN
```

目录里至少要有：

```text
configs/server.local.json
data/sift100w/sift_base.fvecs
data/sift100w/sift_query.fvecs
data/sift100w/sift_groundtruth.ivecs
data/index/full/full_hnsw.bin
data/index/full/full_hnsw.meta.json
src/
tests/hnsw/run_queries.py
scripts/run_queries.sh
requirements.txt
```

## 4. 创建 Python 环境

```bash
cd /home/admin/FaasANN
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

如果 `hnswlib` 或 `faiss-cpu` 安装失败，先装编译工具：

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev
```

## 5. 检查索引和 raw vectors 是否匹配

```bash
cat data/index/full/full_hnsw.meta.json
```

应看到：

```json
{
  "vector_count": 1000000,
  "dimension": 128,
  "partitioned": false
}
```

如果 ECS 上没有索引，可以重新构建：

```bash
.venv/bin/python scripts/build_index.py \
  --base data/sift100w/sift_base.fvecs \
  --out-dir data/index/full \
  --topk 1000000 \
  --m 32 \
  --ef-construction 200 \
  --ef-search 80
```

## 6. 修改监听地址

如果需要外部访问，修改：

```text
configs/server.local.json
```

把：

```json
"host": "127.0.0.1"
```

改成：

```json
"host": "0.0.0.0"
```

如果只在 ECS 本机跑 `run_queries.sh`，不用改。

## 7. 启动服务器

```bash
cd /home/admin/FaasANN
source .venv/bin/activate
python src/main.py --config configs/server.local.json
```

预期日志：

```text
loading vectors from ...
loaded hnswlib index from ...
server initialized: vectors=1000000 dim=128 index_backend=hnswlib
```

启动第一次会比较慢，因为要加载 100 万 raw vectors 和 HNSW 索引。

## 8. 测试服务

本机测试：

```bash
curl http://127.0.0.1:8080/health
```

预期：

```json
{"status":"ok","vectors":1000000,"dimension":128}
```

单 query 调试：

```bash
curl -X POST http://127.0.0.1:8080/search \
  -H 'content-type: application/json' \
  -d '{"query_id": 0, "k": 10}'
```

正式 recall 测试：

```bash
cd /home/admin/FaasANN/scripts
./run_queries.sh \
  --query-num 10000 \
  --concurrent-requests 100
```

结果写到：

```text
logs/run_queries.csv
```

## 9. 后台运行

临时方式：

```bash
nohup .venv/bin/python src/main.py --config configs/server.local.json > server.log 2>&1 &
```

更规范的方式是写 systemd service。

示例：

```ini
[Unit]
Description=FaasANN Server
After=network.target

[Service]
WorkingDirectory=/home/admin/FaasANN
ExecStart=/home/admin/FaasANN/.venv/bin/python /home/admin/FaasANN/src/main.py --config /home/admin/FaasANN/configs/server.local.json
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

保存为：

```text
/etc/systemd/system/faasann.service
```

启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable faasann
sudo systemctl start faasann
sudo systemctl status faasann
```

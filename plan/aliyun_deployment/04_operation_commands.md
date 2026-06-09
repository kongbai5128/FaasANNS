# 04. 常用操作命令

以下命令假设项目在：

```text
/home/admin/FaasANN
```

如果你的 ECS 用户目录不同，请替换路径。

## 安装依赖

```bash
cd /home/admin/FaasANN
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 构建完整 HNSW 索引

```bash
cd /home/admin/FaasANN
source .venv/bin/activate

python scripts/build_index.py \
  --base data/sift100w/sift_base.fvecs \
  --out-dir data/index/full \
  --topk 1000000 \
  --m 32 \
  --ef-construction 200 \
  --ef-search 80
```

## 启动服务器

```bash
cd /home/admin/FaasANN
source .venv/bin/activate
python src/main.py --config configs/server.local.json
```

## 健康检查

```bash
curl http://127.0.0.1:8080/health
```

## 单 query 调试

```bash
curl -X POST http://127.0.0.1:8080/search \
  -H 'content-type: application/json' \
  -d '{"query_id": 0, "k": 10}'
```

## 真实 SIFT 查询评测

```bash
cd /home/admin/FaasANN/scripts
./run_queries.sh \
  --query-num 10000 \
  --concurrent-requests 100
```

## 强制本地路径

```bash
./run_queries.sh \
  --query-num 1000 \
  --concurrent-requests 50 \
  --use-local
```

## 强制 FaaS 路径

```bash
./run_queries.sh \
  --query-num 1000 \
  --concurrent-requests 50 \
  --use-faas
```

## 查看压测日志

```bash
tail -n 5 /home/admin/FaasANN/logs/run_queries.csv
```

## 查看服务器统计

```bash
curl http://127.0.0.1:8080/stats
```

## 后台启动

```bash
cd /home/admin/FaasANN
nohup .venv/bin/python src/main.py --config configs/server.local.json > server.log 2>&1 &
```

## 停止后台服务

```bash
ps aux | grep 'src/main.py'
kill <pid>
```

## 修改为阿里云函数计算 provider

编辑：

```text
configs/server.local.json
```

将：

```json
"faas": {
  "provider": "local"
}
```

改成：

```json
"faas": {
  "provider": "aliyun_http",
  "invoke_timeout_seconds": 10.0,
  "warmup_timeout_seconds": 5.0,
  "endpoints": {
    "default": "https://your-fc-http-trigger"
  }
}
```

然后重启服务器。

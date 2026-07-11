# to_FC: Client -> FC -> VM -> Client

这个目录实现正确的回调链路：

```text
Client -> FC Candidate Search -> VM Rerank Server -> Client
```

注意它不是旧的错误链路：

```text
Client -> FC Candidate Search -> VM Server -> FC Candidate Search -> Client
```

## 目录

- `function/`: 上传到阿里云 FC 的候选召回函数。函数只做 HNSW-PQ candidate search，然后把候选发给 VM。
- `server/rerank_callback_server.py`: VM 上运行的 rerank 服务。它接收 FC 发来的候选，使用 raw vectors 精排，然后直接 POST 到客户端 callback URL。
- `test/run_queries.py`: 客户端压测脚本。它本地启动 callback HTTP server，先请求 FC，然后等待 VM 直接回调最终 top-k。
- `run_queries.sh`: GIST 默认参数的测试入口。
- `build_fc_package.sh`: 打包 `function/` 到 `dist/to_FC_candidate_callback.zip`。

## 启动 VM rerank server

```bash
cd /home/qian/Code/FaasANNS
source .venv/bin/activate
python to_FC/server/rerank_callback_server.py gist --port 8081
```

如果要从云函数访问 VM，需要把 `--port` 对应端口暴露出来，并把 `to_FC/run_queries.sh`
里的 `RERANK_SERVER_URL` 改成 VM 公网地址。

## 启动/部署 FC

本地测试：

```bash
cd /home/qian/Code/FaasANNS/to_FC/function
python3 app.py gist
```

打包上传：

```bash
cd /home/qian/Code/FaasANNS
./to_FC/build_fc_package.sh
```

云函数启动命令：

```bash
python3 app.py gist
```

如果云函数启动命令没有带 `gist`，默认会加载 `sift100w`，客户端发送 GIST 960 维 query 时会报：

```text
query dimension must be 128, got (960,)
```

## 运行测试

本地三段都在本机时：

```bash
cd /home/qian/Code/FaasANNS
./to_FC/run_queries.sh --dataset gist
```

云函数 + VM 真实测试时：

直接改 `to_FC/run_queries.sh` 里的这几行即可：

```bash
FUNCTION_URL="https://to-fc-ncbfswtjlz.cn-hongkong.fcapp.run"
RERANK_SERVER_URL="http://127.0.0.1:8081"
CALLBACK_ADVERTISE_HOST="127.0.0.1"
CALLBACK_PORT="18080"
```

客户端必须能被 VM 访问到，因为最终结果由 VM 直接 POST 到：

```text
http://<CALLBACK_ADVERTISE_HOST>:<CALLBACK_PORT>/callback
```

## CSV 字段

默认输出：

```text
logs/run_queries_to_FC_<dataset>.csv
```

关键字段：

- `qps_client_final_result`: 客户端收到 VM 最终回调结果的 QPS。
- `avg_fc_accept_request_ms`: 客户端请求 FC 并拿到 FC accepted 响应的平均耗时，不是最终结果耗时。
- `avg_client_final_result_ms`: 从客户端请求 FC 到收到 VM 最终回调结果的平均耗时。
- `avg_function_handler_ms`: FC handler 总耗时。
- `avg_function_ann_search_ms`: FC 内 Faiss HNSW-PQ 搜索耗时。
- `avg_function_to_vm_accept_ms`: FC 把候选 POST 给 VM `/rerank_callback` 并收到 accepted 的耗时。
- `avg_server_rerank_ms`: VM exact rerank 耗时。
- `avg_server_total_before_callback_ms`: VM 收到候选后，到发起 callback 前的总耗时。

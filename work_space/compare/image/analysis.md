测试提供http发送1000个query，最高并发度为100。

在保证Recall@10为0.99的情况下，测得平均每个query时延和pqs如下图。

![](https://mcny8yttcq1m.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDAzZTQ0NjQ4OGI5NjExZjkyMjc2ODkyYThlOTBjYzBfQ2o3bkhWd2F4RGtwRngxeHh5cFVwYUM4N2hRalRLTG1fVG9rZW46SU05SmJRZWFQb29MZlJ4a25Bb2M1cXNpbmRnXzE3ODIwMjEyNzI6MTc4MjAyNDg3Ml9WNA&add_watermark=true&scene_type=CCM)

![](https://mcny8yttcq1m.feishu.cn/space/api/box/stream/download/asynccode/?code=NGVkY2JiMTU4NDk3ZDA4ZDFiYmNiZDE1Mzc4ZGE2NDZfeFhxU0Z2SEJTMDE4bktWMmpWblFGZVB3M3V3R3RVTm9fVG9rZW46RldEUGJRT2xMb0pSRmx4dzNIamM5VGE2bm5oXzE3ODIwMjEyNzI6MTc4MjAyNDg3Ml9WNA&add_watermark=true&scene_type=CCM)

保证召回率为0.99时，3种方法的候选集参数

Full Search：

```Plain
SIFT100W ef_search：99
GIST ef_search：250
```

Two-stage：

```Plain
SIFT100W ef_search：221
GIST ef_search：550
```

FaasANN：

```Plain
SIFT100W ef_search：221
GIST ef_search：550
```

1. 整体看来FaasANN在减少冷启动时延有较大作用

```
原因：
```


```Plain
FaasANN 函数端只加载 PQ 候选索引。函数还未启动时，云服务器可以先处理query
Full Search 函数端加载完整 HNSW。
Two-stage 函数端加载 PQ 索引，还要访问 raw vectors。
```

2. 在预热完成后，比较HNSW和使用pq的HNSW的两阶段搜索方法，可以看到两者差距不大，甚至使用pq的要慢一些，即使换成高维数据集后仍然没有较大差距。

   原因：

```Plain
（1）预热后没有数据加载耗时，主要看搜索和远程调用。

Full Search / SIFT100W / warm：
avg_client_ms：342.623 ms
avg_function_handler_ms：1.235 ms
avg_function_hnsw_query_ms：1.154 ms
cold_start_num：0

FaasANN / GIST / warm / 并发100：
avg_total_ms：2999.710 ms
avg_function_handler_ms：9.277 ms
avg_function_faiss_search_ms：8.802 ms
avg_rerank_ms：4.569 ms
avg_remote_invoke_ms：3070.704 ms
avg_remote_queue_estimate_ms：3061.427 ms
cold_start_num：0

可以看到，Full Search 的函数内部 HNSW 搜索只有 1.154 ms。
FaasANN 的函数内部 PQ-HNSW 搜索是 8.802 ms。
PQ-HNSW 并没有比 full HNSW 的这次测试更快。
```

```Plain
（2）PQ-HNSW为了保证0.99召回率，需要更大的ef_search和候选集。

Full Search / SIFT100W：
candidate_k：10
ef_search：99

FaasANN / GIST：
candidate_k：550
ef_search：550

Full Search只需要返回top10。
FaasANN需要先召回550个候选，再做精排。
候选集变大后，PQ压缩带来的搜索优势会被抵消。
```

```Plain
（3）GIST维度更高，精排和候选处理更重。

SIFT100W维度：128
GIST维度：960

即使PQ索引本身更小，最终仍然要处理GIST的高维query和候选结果。
FaasANN还要把候选id从云函数返回给VM，再由VM做rerank。
所以端到端不一定比Full Search快。
```

```Plain
（4）当前GIST主要瓶颈不是函数内部搜索，而是远程调用。

FaasANN / GIST / 并发100：
avg_function_handler_ms：9.277 ms
avg_remote_invoke_ms：3070.704 ms
avg_remote_queue_estimate_ms：3061.427 ms

函数真正执行搜索只用了约9 ms。
但是一次远程调用平均超过3000 ms。
所以总体时延主要被云函数调用、调度、排队、HTTP链路影响。
这时即使PQ搜索更轻，也很难明显反映到总时延上。
```

```Plain
（5）降低并发后，GIST时延明显下降，说明远程调用链路影响很大。

FaasANN / GIST / 并发100：
avg_total_ms：2999.710 ms
qps：17.95

FaasANN / GIST / 并发30：
avg_total_ms：884.414 ms
qps：32.19

并发从100降到30后，平均时延从2999.710 ms降到884.414 ms。
这说明高并发下存在明显的远程调用等待。
瓶颈不是单纯的PQ搜索计算。
```

```Plain
总结：

预热完成后，Full Search已经不需要加载索引。
它的函数内部HNSW搜索本身非常快。

PQ-HNSW虽然减少了索引大小，但为了保证0.99召回率，需要更大的ef_search和候选集。
FaasANN还多了VM到云函数的远程调用和VM侧精排。

所以在端到端结果上，PQ-HNSW两阶段方法没有明显快于Full Search。
在GIST这种高维数据集上，远程调用和候选处理成本更明显，因此差距仍然不大，甚至会更慢。
```

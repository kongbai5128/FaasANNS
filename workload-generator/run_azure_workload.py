import time
import pickle
import requests
import numpy as np
import sys
import json

# 引入 common 以便能反序列化 plan.bin 中的类
from common import dump_t, parameters_t

# 配置
PLAN_FILE = 'plan.bin' # 你的计划文件路径
FUNC_HOST = 'http://localhost:7071' # 请确认你的 func start 端口
ORCHESTRATOR_NAME = 'DurableFunctionsOrchestrator1'
VECTOR_DIM = 128 # 你的向量维度

def main():
    # 1. 加载计划
    print(f"Loading plan from {PLAN_FILE}...")
    try:
        with open(PLAN_FILE, 'rb') as f:
            dump = pickle.load(f)
            plan_times = dump.plan # 这是一个时间戳列表（秒）
            print(f"Loaded {len(plan_times)} tasks.")
    except FileNotFoundError:
        print("Error: plan.bin not found. Run main.py first.")
        return

    # 2. 启动 Orchestrator 实例
    start_url = f"{FUNC_HOST}/api/orchestrators/{ORCHESTRATOR_NAME}"
    print(f"Starting orchestrator instance at {start_url}...")
    
    try:
        resp = requests.post(start_url, json=None)
        resp.raise_for_status()
        info = resp.json()
        instance_id = info['id']
        # 获取发送事件的 URL 模板
        # 通常格式: http://localhost:7071/runtime/webhooks/durabletask/instances/{instanceId}/raiseEvent/{eventName}
        # 我们手动构建以防返回的 URL 是云端格式
        raise_event_url = f"{FUNC_HOST}/runtime/webhooks/durabletask/instances/{instance_id}/raiseEvent/Query"
        
        print(f"Instance started. ID: {instance_id}")
        print(f"Target Event URL: {raise_event_url}")
    except Exception as e:
        print(f"Failed to start orchestrator: {e}")
        return

    # 3. 执行负载
    print("Starting workload execution...")
    start_time = time.perf_counter()
    
    # 这里的 plan_times 是相对于 0 的时间点列表
    # 我们需要根据当前时间来调度
    
    for i, scheduled_time in enumerate(plan_times):
        # 计算需要等待的时间
        current_elapsed = time.perf_counter() - start_time
        wait_time = scheduled_time - current_elapsed
        
        if wait_time > 0:
            time.sleep(wait_time)
            
        # 构造一个随机向量作为 Query (模拟)
        # 注意：你的 Orchestrator 期望接收一个能被 np.linalg.norm 计算的数组
        query_vector = np.random.rand(VECTOR_DIM).tolist()
        
        try:
            # 发送 'Query' 事件
            # header Content-Type: application/json 是必须的
            r = requests.post(raise_event_url, json=query_vector)
            if r.status_code == 202:
                print(f"[{i+1}/{len(plan_times)}] Sent query at t={scheduled_time:.2f}s")
            else:
                print(f"[{i+1}/{len(plan_times)}] Failed to send: {r.status_code} {r.text}")
        except Exception as e:
            print(f"Error sending request: {e}")

    print("Workload finished.")

if __name__ == "__main__":
    main()
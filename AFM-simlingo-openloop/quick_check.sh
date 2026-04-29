#!/bin/bash
echo "======================================"
echo "快速诊断 - $(date)"
echo "======================================"

echo -e "\n[1] 完成的routes:"
COMPLETED=$(ls -1 eval_results/Bench2Drive/simlingo/bench2drive/1/res/*.json 2>/dev/null | wc -l)
echo "  总数: $COMPLETED / 220"

echo -e "\n[2] 正在运行的评估进程:"
RUNNING=$(ps aux | grep leaderboard_evaluator | grep -v grep | wc -l)
echo "  数量: $RUNNING"
if [ $RUNNING -gt 0 ]; then
    ps aux | grep leaderboard_evaluator | grep -v grep | awk '{print "  PID:", $2, "运行时间:", $10}'
fi

echo -e "\n[3] Python主脚本状态:"
ps aux | grep "start_eval_simlingo_gpu.py" | grep -v grep

echo -e "\n[4] 最近的日志文件:"
ls -lt eval_results/Bench2Drive/simlingo/bench2drive/1/out/*.log 2>/dev/null | head -3 | awk '{print "  ", $9, $6, $7, $8}'

echo -e "\n[5] CARLA进程:"
ps aux | grep CarlaUE4 | grep -v grep | wc -l

echo -e "\n[6] GPU使用:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

echo "======================================"

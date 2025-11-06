#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Webhook数据格式
展示发送到webhook的JSON数据格式
"""

import json
import numpy as np
from datetime import datetime

def create_sample_webhook_data():
    """创建样本webhook数据"""

    # 模拟窗口数据统计
    sensor_stats = {
        "T24": {"min": 641.82, "max": 643.49, "mean": 642.65, "std": 0.45},
        "T30": {"min": 1582.79, "max": 1605.26, "mean": 1590.52, "std": 5.21},
        "T50": {"min": 1400.60, "max": 1433.58, "mean": 1415.89, "std": 8.34},
        "P30": {"min": 553.75, "max": 556.36, "mean": 554.89, "std": 0.67},
        "Nf": {"min": 2382.04, "max": 2388.11, "mean": 2385.67, "std": 1.23},
        "Nc": {"min": 9044.07, "max": 9072.94, "mean": 9058.45, "std": 6.78},
        "Ps30": {"min": 47.06, "max": 47.49, "mean": 47.28, "std": 0.12},
        "phi": {"min": 521.66, "max": 525.89, "mean": 523.45, "std": 1.02},
        "NRf": {"min": 2382.03, "max": 2388.08, "mean": 2385.34, "std": 1.19},
        "BPR": {"min": 8.37, "max": 8.50, "mean": 8.44, "std": 0.03},
        "htBleed": {"min": 388, "max": 400, "mean": 394.5, "std": 2.8},
        "W31": {"min": 38.14, "max": 39.43, "mean": 38.82, "std": 0.28},
        "W32": {"min": 22.89, "max": 23.62, "mean": 23.29, "std": 0.15}
    }

    webhook_data = {
        "timestamp": datetime.now().isoformat(),
        "engine_id": 3,
        "prediction": {
            "predicted_rul": 99.21,
            "window_start_cycle": 1,
            "window_end_cycle": 50,
            "window_size": 50,
            "processing_time_seconds": 0.002
        },
        "engine_info": {
            "total_cycles": 126,
            "current_position": 1,
            "total_predictions": 1
        },
        "model_info": {
            "model_file": "xgboost_rul_model_20251105_121121.pkl",
            "prediction_count": 1
        },
        "simulation_info": {
            "simulation_start_time": "2025-11-05T14:41:18.164893",
            "webhook_success_count": 0,
            "webhook_failure_count": 1
        },
        "sensor_statistics": sensor_stats
    }

    return webhook_data

def main():
    """主函数"""
    print("=== Webhook数据格式示例 ===\n")

    # 创建样本数据
    webhook_data = create_sample_webhook_data()

    # 打印JSON格式
    print("发送到Webhook的JSON数据格式:")
    print("=" * 50)
    print(json.dumps(webhook_data, indent=2, ensure_ascii=False))

    print("\n" + "=" * 50)
    print(f"数据大小: {len(json.dumps(webhook_data))} 字符")
    print(f"字段数量: {len(webhook_data)} 个主要字段")
    print(f"传感器统计: {len(webhook_data['sensor_statistics'])} 个传感器")

    # 显示主要字段说明
    print("\n=== 主要字段说明 ===")
    print("1. timestamp: 预测时间戳 (ISO 8601格式)")
    print("2. engine_id: 发动机ID")
    print("3. prediction: 预测结果信息")
    print("   - predicted_rul: 预测的RUL值")
    print("   - window_start_cycle/end_cycle: 窗口周期范围")
    print("   - processing_time_seconds: 处理时间")
    print("4. engine_info: 发动机相关信息")
    print("5. model_info: 模型相关信息")
    print("6. simulation_info: 模拟运行相关信息")
    print("7. sensor_statistics: 传感器数据统计信息")

if __name__ == "__main__":
    main()
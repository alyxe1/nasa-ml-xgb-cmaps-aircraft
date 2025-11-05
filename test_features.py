#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型特征一致性
"""

import pandas as pd
import numpy as np
import joblib

def test_feature_consistency():
    """测试训练和预测的特征一致性"""
    print("=== 测试特征一致性 ===")

    # 加载模型参数
    window_params = joblib.load('saved_model/window_params_20251105_120718.pkl')
    expected_features = window_params['feature_columns']
    print(f"模型期望特征: {expected_features}")
    print(f"模型期望特征数: {len(expected_features)}")

    # 加载并预处理测试数据（与训练脚本完全一致）
    test_df = pd.read_csv('CMaps/test_FD001.txt', sep='\s+', header=None)
    columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
             [f'sensor{i}' for i in range(1, 22)]
    test_df.columns = columns

    # 删除空列
    test_df.drop(columns=[26, 27], inplace=True, errors='ignore')

    # 删除常数列（与训练脚本一致）
    columns_to_drop = ['sensor1', 'sensor6', 'sensor10', 'sensor16', 'sensor18', 'sensor19']
    test_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # 传感器名称映射（与训练脚本一致）
    sensor_mapping = {
        'sensor2': 'T24', 'sensor3': 'T30', 'sensor4': 'T50', 'sensor7': 'P30',
        'sensor8': 'Nf', 'sensor9': 'Nc', 'sensor11': 'Ps30', 'sensor12': 'phi',
        'sensor13': 'NRf', 'sensor14': 'BPR', 'sensor15': 'htBleed', 'sensor17': 'W31',
        'sensor20': 'W32', 'sensor21': 'RUL'
    }

    for old_name, new_name in sensor_mapping.items():
        if old_name in test_df.columns:
            test_df.rename(columns={old_name: new_name}, inplace=True)

    print(f"预处理后测试数据列: {test_df.columns.tolist()}")

    # 检查特征列是否匹配
    available_features = [col for col in expected_features if col in test_df.columns]
    missing_features = [col for col in expected_features if col not in test_df.columns]
    extra_features = [col for col in test_df.columns if col not in expected_features + ['unit_number', 'time_in_cycles', 'RUL']]

    print(f"可用特征: {available_features}")
    print(f"缺失特征: {missing_features}")
    print(f"多余特征: {extra_features}")

    # 创建一个测试样本
    if len(missing_features) == 0:
        print("✅ 特征完全匹配！")

        # 选择发动机3的数据
        engine_3_data = test_df[test_df['unit_number'] == 3].copy()
        if len(engine_3_data) >= 50:
            # 加载RUL数据
            rul_df = pd.read_csv('CMaps/RUL_FD001.txt', sep='\s+', header=None)
            rul_df.columns = ['RUL']
            rul_df.index = range(1, len(rul_df) + 1)

            # 添加RUL
            unit_rul = rul_df.loc[3, 'RUL']
            engine_3_data = engine_3_data.sort_values('time_in_cycles')
            max_cycles = engine_3_data['time_in_cycles'].max()
            engine_3_data['RUL'] = unit_rul + max_cycles - engine_3_data['time_in_cycles']

            # 创建滑动窗口
            window_size = 50
            feature_data = engine_3_data[expected_features].values
            window_features = feature_data[:window_size].flatten()

            print(f"滑动窗口特征维度: {len(window_features)}")
            print(f"预期特征维度: {len(expected_features) * window_size}")

            if len(window_features) == len(expected_features) * window_size:
                print("✅ 特征维度完全匹配！")
                return True
            else:
                print("❌ 特征维度不匹配")
                return False
        else:
            print(f"❌ 发动机3数据不足: {len(engine_3_data)} < {window_size}")
            return False
    else:
        print("❌ 特征不匹配")
        return False

if __name__ == "__main__":
    test_feature_consistency()
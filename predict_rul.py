#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUL预测推理脚本

使用训练好的XGBoost模型进行RUL预测

作者: Claude Code Assistant
日期: 2025-11-04
"""

import os
import sys
import warnings
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

class RULPredictor:
    """RUL预测器"""

    def __init__(self, model_path=None, metadata_path=None, params_path=None):
        """
        初始化预测器

        Args:
            model_path (str): 模型文件路径
            metadata_path (str): 元数据文件路径
            params_path (str): 参数文件路径
        """
        self.model = None
        self.metadata = None
        self.window_params = None

        # 加载模型和相关文件
        self.load_model(model_path, metadata_path, params_path)

    def load_model(self, model_path=None, metadata_path=None, params_path=None):
        """
        加载训练好的模型

        Args:
            model_path (str): 模型文件路径
            metadata_path (str): 元数据文件路径
            params_path (str): 参数文件路径
        """
        print("=== 加载模型 ===")

        # 确定模型路径
        if model_path is None:
            # 尝试使用最新模型
            model_path = "saved_model/latest_model.pkl"
            if not os.path.exists(model_path):
                # 如果符号链接不存在，查找最新的模型文件
                saved_dir = Path("saved_model")
                model_files = list(saved_dir.glob("xgboost_rul_model_*.pkl"))
                if model_files:
                    model_path = str(max(model_files, key=os.path.getctime))
                else:
                    raise FileNotFoundError("未找到训练好的模型文件")

        print(f"模型文件: {model_path}")

        # 加载模型
        self.model = joblib.load(model_path)
        print(f"模型加载成功")
        print(f"模型特征数: {self.model.get_booster().num_features()}")

        # 加载元数据
        if metadata_path is None:
            # 从模型路径推断元数据路径
            model_name = os.path.basename(model_path)
            timestamp = model_name.split('_')[-1].replace('.pkl', '')
            metadata_path = f"saved_model/model_metadata_{timestamp}.json"

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"元数据加载成功")
            print(f"窗口大小: {self.metadata['window_size']}")
            print(f"训练日期: {self.metadata['training_date']}")
        else:
            print(f"警告: 元数据文件不存在: {metadata_path}")

        # 加载窗口参数
        if params_path is None:
            # 从模型路径推断参数路径
            model_name = os.path.basename(model_path)
            timestamp = model_name.split('_')[-1].replace('.pkl', '')
            params_path = f"saved_model/window_params_{timestamp}.pkl"

        if os.path.exists(params_path):
            self.window_params = joblib.load(params_path)
            print(f"窗口参数加载成功")
            print(f"特征列数: {len(self.window_params['feature_columns'])}")
        else:
            print(f"警告: 参数文件不存在: {params_path}")

    def create_sliding_windows(self, data, window_size=None, step_size=1):
        """
        创建滑动窗口数据

        Args:
            data (pd.DataFrame): 包含发动机数据的DataFrame
            window_size (int): 窗口大小
            step_size (int): 滑动步长

        Returns:
            tuple: (sequences, metadata)
        """
        if window_size is None:
            window_size = self.metadata['window_size'] if self.metadata else 50

        sequences = []
        metadata = []

        # 获取特征列
        if self.window_params and 'feature_columns' in self.window_params:
            feature_columns = self.window_params['feature_columns']
        else:
            # 使用所有非元数据列
            feature_columns = [col for col in data.columns
                             if col not in ['unit_number', 'time_in_cycles', 'RUL']]

        for unit in sorted(data['unit_number'].unique()):
            unit_data = data[data['unit_number'] == unit].copy()
            unit_data = unit_data.sort_values('time_in_cycles')

            # 检查数据是否足够
            if len(unit_data) < window_size:
                print(f"警告: 发动机 {unit} 数据点不足 ({len(unit_data)} < {window_size})，跳过")
                continue

            # 获取特征值
            unit_features = unit_data[feature_columns].values

            # 创建滑动窗口
            for i in range(0, len(unit_data) - window_size + 1, step_size):
                window_features = unit_features[i:i + window_size].flatten()
                sequences.append(window_features)

                # 保存元数据
                metadata.append({
                    'unit_number': unit,
                    'window_start_idx': i,
                    'window_end_idx': i + window_size - 1,
                    'window_start_cycle': unit_data.iloc[i]['time_in_cycles'],
                    'window_end_cycle': unit_data.iloc[i + window_size - 1]['time_in_cycles']
                })

        return np.array(sequences), metadata

    def predict_rul(self, data, window_size=None, batch_size=1000):
        """
        预测RUL

        Args:
            data (pd.DataFrame): 发动机数据
            window_size (int): 窗口大小
            batch_size (int): 批处理大小

        Returns:
            tuple: (predictions, metadata)
        """
        print(f"\n=== RUL预测 ===")
        print(f"数据形状: {data.shape}")

        if window_size is None:
            window_size = self.metadata['window_size'] if self.metadata else 50

        # 创建滑动窗口
        print("创建滑动窗口...")
        sequences, metadata = self.create_sliding_windows(data, window_size=window_size)
        print(f"创建窗口数: {len(sequences)}")
        print(f"每个窗口特征数: {sequences[0].shape if len(sequences) > 0 else 'N/A'}")

        # 检查特征维度
        expected_features = sequences[0].shape[0] if len(sequences) > 0 else 0
        model_features = self.model.get_booster().num_features()

        print(f"数据特征维度: {expected_features}")
        print(f"模型特征维度: {model_features}")

        if expected_features != model_features:
            print(f"警告: 特征维度不匹配")
            if expected_features > model_features:
                sequences = sequences[:, :model_features]
                print(f"截断到 {model_features} 维")
            else:
                raise ValueError(f"数据特征维度({expected_features})小于模型特征维度({model_features})")

        # 分批预测
        print(f"开始预测，总样本数: {len(sequences)}")
        predictions = []

        for i in range(0, len(sequences), batch_size):
            batch_end = min(i + batch_size, len(sequences))
            batch_data = sequences[i:batch_end]

            print(f"预测进度: {i+1}-{batch_end}/{len(sequences)} ({batch_end/len(sequences)*100:.1f}%)")

            batch_pred = self.model.predict(batch_data)
            predictions.extend(batch_pred)

        predictions = np.array(predictions)

        print(f"预测完成!")
        print(f"预测结果形状: {predictions.shape}")
        print(f"预测RUL范围: {predictions.min():.1f} - {predictions.max():.1f}")
        print(f"预测RUL均值: {predictions.mean():.1f}")

        return predictions, metadata

    def predict_engine_rul(self, data, engine_id):
        """
        预测单个发动机的RUL

        Args:
            data (pd.DataFrame): 发动机数据
            engine_id (int): 发动机ID

        Returns:
            dict: 预测结果
        """
        print(f"\n=== 预测发动机 {engine_id} RUL ===")

        # 筛选特定发动机数据
        engine_data = data[data['unit_number'] == engine_id].copy()

        if len(engine_data) == 0:
            raise ValueError(f"未找到发动机 {engine_id} 的数据")

        engine_data = engine_data.sort_values('time_in_cycles')
        print(f"发动机 {engine_id} 数据点: {len(engine_data)}")

        # 预测
        predictions, metadata = self.predict_rul(engine_data)

        # 返回最后一个预测（最新周期的RUL预测）
        if len(predictions) > 0:
            latest_prediction = predictions[-1]
            latest_cycle = engine_data['time_in_cycles'].max()

            result = {
                'engine_id': engine_id,
                'latest_cycle': latest_cycle,
                'predicted_rul': float(latest_prediction),
                'all_predictions': predictions.tolist(),
                'metadata': metadata
            }

            print(f"发动机 {engine_id} 在周期 {latest_cycle} 的预测RUL: {latest_prediction:.1f}")
            return result
        else:
            print(f"发动机 {engine_id} 数据不足以进行预测")
            return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RUL预测脚本')
    parser.add_argument('--model-path', help='模型文件路径')
    parser.add_argument('--metadata-path', help='元数据文件路径')
    parser.add_argument('--params-path', help='参数文件路径')
    parser.add_argument('--data-file', help='预测数据文件路径')
    parser.add_argument('--engine-id', type=int, help='预测特定发动机ID')
    parser.add_argument('--window-size', type=int, help='滑动窗口大小')
    parser.add_argument('--batch-size', type=int, default=1000, help='批处理大小')

    args = parser.parse_args()

    print("RUL预测推理脚本")
    print("=" * 40)

    # 创建预测器
    predictor = RULPredictor(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        params_path=args.params_path
    )

    # 如果没有提供数据文件，使用测试数据进行演示
    if args.data_file is None:
        print("未指定数据文件，使用测试数据进行演示...")

        # 加载测试数据
        test_df = pd.read_csv('CMaps/test_FD001.txt', sep='\s+', header=None)
        columns = ['unit_number', 'time_in_cycles', 'operational_setting_1',
                  'operational_setting_2', 'operational_setting_3'] + \
                 [f'sensor_{i}' for i in range(1, 22)]
        test_df.columns = columns

        # 加载RUL数据
        rul_df = pd.read_csv('CMaps/RUL_FD001.txt', sep='\s+', header=None)
        rul_df.columns = ['RUL']

        # 准备测试数据
        test_with_rul = []
        for unit in test_df['unit_number'].unique():
            unit_test_data = test_df[test_df['unit_number'] == unit].copy()
            unit_rul = rul_df.loc[unit - 1, 'RUL']

            unit_test_data = unit_test_data.sort_values('time_in_cycles')
            max_cycles = unit_test_data['time_in_cycles'].max()
            unit_test_data['RUL'] = unit_rul + max_cycles - unit_test_data['time_in_cycles']

            test_with_rul.append(unit_test_data)

        test_data = pd.concat(test_with_rul, ignore_index=True)

        # 如果指定了发动机ID，只预测该发动机
        if args.engine_id:
            result = predictor.predict_engine_rul(test_data, args.engine_id)
            if result:
                print(f"\n预测结果:")
                print(f"发动机ID: {result['engine_id']}")
                print(f"最新周期: {result['latest_cycle']}")
                print(f"预测RUL: {result['predicted_rul']:.1f}")
        else:
            # 预测所有发动机
            predictions, metadata = predictor.predict_rul(
                test_data,
                window_size=args.window_size,
                batch_size=args.batch_size
            )

            print(f"\n总预测结果:")
            print(f"预测样本数: {len(predictions)}")
            print(f"RUL范围: {predictions.min():.1f} - {predictions.max():.1f}")
            print(f"平均RUL: {predictions.mean():.1f}")

    else:
        # 使用指定的数据文件
        print(f"加载数据文件: {args.data_file}")
        data = pd.read_csv(args.data_file)

        if args.engine_id:
            result = predictor.predict_engine_rul(data, args.engine_id)
            if result:
                print(f"\n预测结果: {result}")
        else:
            predictions, metadata = predictor.predict_rul(
                data,
                window_size=args.window_size,
                batch_size=args.batch_size
            )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产级XGBoost滑动窗口RUL预测模型训练脚本

基于 xgboost-sliding-window-prediction-cn.ipynb 的生产实现
快速实现数据读取、处理、训练和模型保存

作者: Eddy
日期: 2025-11-04
"""

import os
import sys
import warnings
import argparse
import json
import joblib
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

class XGBoostRULTrainer:
    """XGBoost RUL预测模型训练器"""

    def __init__(self, data_dir="CMaps", window_size=50, random_state=42):
        """
        初始化训练器

        Args:
            data_dir (str): 数据目录路径
            window_size (int): 滑动窗口大小
            random_state (int): 随机种子
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.random_state = random_state
        self.model = None
        self.metadata = {}

        # 重要传感器配置（与notebook保持一致）
        self.sensor_cols = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'BPR', 'htBleed', 'W31', 'W32']
        self.operational_cols = ['setting_1', 'setting_2']
        self.feature_cols = self.operational_cols + self.sensor_cols

        print(f"初始化XGBoost RUL训练器:")
        print(f"  数据目录: {self.data_dir}")
        print(f"  窗口大小: {self.window_size}")
        print(f"  特征列数: {len(self.feature_cols)}")
        print(f"  传感器数: {len(self.sensor_cols)}")

    def load_data(self):
        """加载NASA C-MAPS数据集"""
        print("\n=== 加载数据 ===")

        # 定义列名
        columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
                 [f'sensor{i}' for i in range(1, 22)]

        # 加载训练数据
        train_path = self.data_dir / 'train_FD001.txt'
        if not train_path.exists():
            raise FileNotFoundError(f"训练数据文件不存在: {train_path}")

        self.train_df = pd.read_csv(train_path, sep='\s+', header=None)
        self.train_df.columns = columns
        print(f"原始训练数据: {self.train_df.shape}")

        # 加载测试数据
        test_path = self.data_dir / 'test_FD001.txt'
        if not test_path.exists():
            raise FileNotFoundError(f"测试数据文件不存在: {test_path}")

        self.test_df = pd.read_csv(test_path, sep='\s+', header=None)
        self.test_df.columns = columns
        print(f"原始测试数据: {self.test_df.shape}")

        # 数据预处理（与notebook保持一致）
        print("执行数据预处理...")

        # 删除空列
        for df in [self.train_df, self.test_df]:
            df.drop(columns=[26, 27], inplace=True, errors='ignore')

        # 删除常数列
        columns_to_drop = ['sensor1', 'sensor6', 'sensor10', 'sensor16', 'sensor18', 'sensor19',
                          'sensor22', 'sensor23', 'sensor24', 'sensor25']
        for df in [self.train_df, self.test_df]:
            df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        # 传感器名称映射（与notebook保持一致）
        sensor_mapping = {
            'sensor2': 'T24', 'sensor3': 'T30', 'sensor4': 'T50', 'sensor7': 'P30',
            'sensor8': 'Nf', 'sensor9': 'Nc', 'sensor11': 'Ps30', 'sensor12': 'phi',
            'sensor13': 'NRf', 'sensor14': 'BPR', 'sensor15': 'htBleed', 'sensor17': 'W31',
            'sensor20': 'W32', 'sensor21': 'RUL'
        }

        for df_name, df in [('train_df', self.train_df), ('test_df', self.test_df)]:
            for old_name, new_name in sensor_mapping.items():
                if old_name in df.columns:
                    df.rename(columns={old_name: new_name}, inplace=True)
            # 更新引用
            if df_name == 'train_df':
                self.train_df = df
            else:
                self.test_df = df

        print(f"预处理后训练数据: {self.train_df.shape}")
        print(f"预处理后测试数据: {self.test_df.shape}")

        # 加载RUL数据
        rul_path = self.data_dir / 'RUL_FD001.txt'
        if not rul_path.exists():
            raise FileNotFoundError(f"RUL数据文件不存在: {rul_path}")

        self.rul_df = pd.read_csv(rul_path, sep='\s+', header=None)
        self.rul_df.columns = ['RUL']
        self.rul_df.index = range(1, len(self.rul_df) + 1)  # 发动机编号从1开始
        print(f"RUL数据: {self.rul_df.shape}")

        return self.train_df, self.test_df, self.rul_df

    def add_rul_to_dataframe(self, df):
        """为DataFrame添加RUL列"""
        df_with_rul = df.copy()
        df_with_rul['RUL'] = df_with_rul.groupby('unit_number')['time_in_cycles'].transform(
            lambda x: x.max() - x
        )
        return df_with_rul

    def create_sliding_windows(self, data, step_size=1):
        """
        创建滑动窗口数据

        Args:
            data (pd.DataFrame): 包含发动机数据的DataFrame
            step_size (int): 滑动步长

        Returns:
            tuple: (sequences, labels, metadata)
        """
        sequences = []
        labels = []
        metadata = []

        # 获取特征列（排除元数据列）
        feature_columns = [col for col in data.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]

        for unit in sorted(data['unit_number'].unique()):
            unit_data = data[data['unit_number'] == unit].copy()
            unit_data = unit_data.sort_values('time_in_cycles')

            # 检查数据是否足够创建窗口
            if len(unit_data) < self.window_size:
                print(f"警告: 发动机 {unit} 数据点不足 ({len(unit_data)} < {self.window_size})，跳过")
                continue

            # 获取特征值
            unit_features = unit_data[feature_columns].values

            # 创建滑动窗口
            for i in range(0, len(unit_data) - self.window_size + 1, step_size):
                # 获取窗口特征并展平
                window_features = unit_features[i:i + self.window_size].flatten()
                sequences.append(window_features)

                # 使用窗口结束时的RUL作为标签
                window_rul = unit_data.iloc[i + self.window_size - 1]['RUL']
                labels.append(window_rul)

                # 保存元数据
                metadata.append({
                    'unit_number': unit,
                    'window_start_idx': i,
                    'window_end_idx': i + self.window_size - 1,
                    'window_start_cycle': unit_data.iloc[i]['time_in_cycles'],
                    'window_end_cycle': unit_data.iloc[i + self.window_size - 1]['time_in_cycles'],
                    'rul': window_rul
                })

        return np.array(sequences), np.array(labels), metadata

    def prepare_data(self):
        """准备训练和测试数据"""
        print("\n=== 准备数据 ===")

        # 添加RUL到训练数据
        self.train_df_with_rul = self.add_rul_to_dataframe(self.train_df)
        print(f"训练数据添加RUL: {self.train_df_with_rul.shape}")

        # 选择重要特征列
        train_features = ['unit_number', 'time_in_cycles', 'RUL'] + self.feature_cols
        self.train_df_filtered = self.train_df_with_rul[train_features]
        print(f"训练数据特征过滤: {self.train_df_filtered.shape}")

        # 创建训练数据滑动窗口
        print("创建训练数据滑动窗口...")
        self.X_train, self.y_train, self.train_metadata = self.create_sliding_windows(
            self.train_df_filtered, step_size=1
        )
        print(f"训练滑动窗口: {self.X_train.shape}, 标签: {self.y_train.shape}")

        # 准备测试数据
        print("准备测试数据...")

        # 为测试数据添加RUL
        self.test_with_rul = []

        for unit in self.test_df['unit_number'].unique():
            unit_test_data = self.test_df[self.test_df['unit_number'] == unit].copy()
            unit_rul = self.rul_df.loc[unit, 'RUL']  # RUL DataFrame索引从1开始

            # 为测试数据的每个时间点计算实际RUL
            unit_test_data = unit_test_data.sort_values('time_in_cycles')
            max_cycles = unit_test_data['time_in_cycles'].max()
            unit_test_data['RUL'] = unit_rul + max_cycles - unit_test_data['time_in_cycles']

            self.test_with_rul.append(unit_test_data)

        self.test_with_rul = pd.concat(self.test_with_rul, ignore_index=True)

        # 选择特征列
        test_features = ['unit_number', 'time_in_cycles', 'RUL'] + self.feature_cols
        self.test_df_filtered = self.test_with_rul[test_features]
        print(f"测试数据特征过滤: {self.test_df_filtered.shape}")

        # 创建测试数据滑动窗口
        print("创建测试数据滑动窗口...")
        self.X_test, self.y_test, self.test_metadata = self.create_sliding_windows(
            self.test_df_filtered, step_size=1
        )
        print(f"测试滑动窗口: {self.X_test.shape}, 标签: {self.y_test.shape}")

        return self.X_train, self.y_train, self.X_test, self.y_test

    def train_model(self, n_estimators=300, learning_rate=0.05, max_depth=8, validation_size=0.2):
        """
        训练XGBoost模型

        Args:
            n_estimators (int): 树的数量
            learning_rate (float): 学习率
            max_depth (int): 树的最大深度
            validation_size (float): 验证集比例
        """
        print("\n=== 训练模型 ===")

        # 分割训练和验证数据
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            self.X_train, self.y_train,
            test_size=validation_size,
            random_state=self.random_state
        )

        print(f"训练集: {X_train_split.shape}")
        print(f"验证集: {X_val_split.shape}")
        print(f"测试集: {self.X_test.shape}")
        print(f"特征维度: {X_train_split.shape[1]} ({X_train_split.shape[1]//self.window_size} × {self.window_size})")

        # 创建XGBoost模型
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            objective='reg:squarederror'
        )

        print(f"\n模型参数:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  max_depth: {max_depth}")

        # 训练模型
        print("\n开始训练...")
        start_time = datetime.now()

        self.model.fit(X_train_split, y_train_split)

        training_time = datetime.now() - start_time
        print(f"训练完成! 耗时: {training_time}")

        # 评估模型
        print("\n=== 模型评估 ===")

        # 训练集评估
        train_pred = self.model.predict(X_train_split)
        train_mae = mean_absolute_error(y_train_split, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_split, train_pred))
        train_r2 = r2_score(y_train_split, train_pred)

        print(f"训练集性能:")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  R2: {train_r2:.4f}")

        # 验证集评估
        val_pred = self.model.predict(X_val_split)
        val_mae = mean_absolute_error(y_val_split, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val_split, val_pred))
        val_r2 = r2_score(y_val_split, val_pred)

        print(f"\n验证集性能:")
        print(f"  MAE: {val_mae:.4f}")
        print(f"  RMSE: {val_rmse:.4f}")
        print(f"  R2: {val_r2:.4f}")

        # 测试集评估
        test_pred = self.model.predict(self.X_test)
        test_mae = mean_absolute_error(self.y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        test_r2 = r2_score(self.y_test, test_pred)

        print(f"\n测试集性能:")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  R2: {test_r2:.4f}")

        # 保存模型元数据
        self.metadata = {
            'model_type': 'XGBoost Regressor',
            'window_size': self.window_size,
            'feature_count': X_train_split.shape[1],
            'feature_columns': self.feature_cols,
            'training_samples': X_train_split.shape[0],
            'validation_samples': X_val_split.shape[0],
            'test_samples': self.X_test.shape[0],
            'model_params': {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'random_state': self.random_state
            },
            'performance': {
                'train_mae': float(train_mae),
                'train_rmse': float(train_rmse),
                'train_r2': float(train_r2),
                'val_mae': float(val_mae),
                'val_rmse': float(val_rmse),
                'val_r2': float(val_r2),
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'test_r2': float(test_r2)
            },
            'training_time': str(training_time),
            'training_date': datetime.now().strftime('%Y%m%d_%H%M%S')
        }

        return self.model

    def save_model(self, save_dir="saved_model"):
        """
        保存训练好的模型

        Args:
            save_dir (str): 保存目录
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先训练模型")

        print("\n=== 保存模型 ===")

        # 创建保存目录
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        print(f"保存目录: {save_path.absolute()}")

        # 生成时间戳
        timestamp = self.metadata['training_date']

        # 保存模型
        model_file = save_path / f"xgboost_rul_model_{timestamp}.pkl"
        joblib.dump(self.model, model_file)
        print(f"模型已保存: {model_file}")

        # 保存元数据
        metadata_file = save_path / f"model_metadata_{timestamp}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"元数据已保存: {metadata_file}")

        # 保存滑动窗口参数
        window_params = {
            'window_size': self.window_size,
            'feature_columns': self.feature_cols,
            'sensor_columns': self.sensor_cols,
            'operational_columns': self.operational_cols
        }
        params_file = save_path / f"window_params_{timestamp}.pkl"
        joblib.dump(window_params, params_file)
        print(f"窗口参数已保存: {params_file}")

        # 创建最新模型符号链接
        latest_model = save_path / "latest_model.pkl"
        if latest_model.exists():
            latest_model.unlink()

        try:
            latest_model.symlink_to(model_file.name)
            print(f"最新模型链接: {latest_model}")
        except OSError:
            # Windows可能需要管理员权限创建符号链接
            print(f"无法创建符号链接，请手动使用: {model_file}")

        return {
            'model_file': str(model_file),
            'metadata_file': str(metadata_file),
            'params_file': str(params_file)
        }

    def train_and_save(self, save_dir="saved_model", **model_params):
        """
        完整的训练和保存流程

        Args:
            save_dir (str): 保存目录
            **model_params: 模型参数
        """
        try:
            # 加载数据
            self.load_data()

            # 准备数据
            self.prepare_data()

            # 训练模型
            self.train_model(**model_params)

            # 保存模型
            saved_files = self.save_model(save_dir)

            print("\n=== 训练完成 ===")
            print(f"模型保存路径: {saved_files['model_file']}")
            print(f"性能指标: MAE={self.metadata['performance']['test_mae']:.4f}")

            return saved_files

        except Exception as e:
            print(f"训练过程中出现错误: {str(e)}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='XGBoost RUL模型训练脚本')
    parser.add_argument('--data-dir', default='CMaps', help='数据目录路径')
    parser.add_argument('--save-dir', default='saved_model', help='模型保存目录')
    parser.add_argument('--window-size', type=int, default=50, help='滑动窗口大小')
    parser.add_argument('--n-estimators', type=int, default=300, help='XGBoost树的数量')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='学习率')
    parser.add_argument('--max-depth', type=int, default=8, help='树的最大深度')
    parser.add_argument('--random-state', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    print("XGBoost RUL预测模型训练脚本")
    print("=" * 50)

    # 创建训练器
    trainer = XGBoostRULTrainer(
        data_dir=args.data_dir,
        window_size=args.window_size,
        random_state=args.random_state
    )

    # 训练参数
    model_params = {
        'n_estimators': args.n_estimators,
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth
    }

    # 执行训练
    saved_files = trainer.train_and_save(
        save_dir=args.save_dir,
        **model_params
    )

    print(f"\n使用方法:")
    print(f"import joblib")
    print(f"model = joblib.load('{saved_files['model_file']}')")
    print(f"metadata = joblib.load('{saved_files['metadata_file']}')")


if __name__ == "__main__":
    main()
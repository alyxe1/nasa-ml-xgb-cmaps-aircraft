#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时发动机RUL预测模拟系统

模拟真实发动机运行，每10分钟滑动窗口向前移动一次，
循环遍历CMaps数据集进行实时RUL预测。

作者: Eddy
日期: 2025-11-04
"""

import os
import sys
import json
import time
import logging
import argparse
import signal
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from croniter import croniter

# 导入现有的预测模块
from predict_rul import RULPredictor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RealtimeEngineSimulator:
    """实时发动机模拟器"""

    def __init__(self, engine_id=1, data_file="CMaps/test_FD001.txt",
                 rul_file="CMaps/RUL_FD001.txt", model_path=None):
        """
        初始化模拟器

        Args:
            engine_id (int): 要模拟的发动机ID
            data_file (str): 测试数据文件路径
            rul_file (str): RUL数据文件路径
            model_path (str): 模型文件路径
        """
        self.engine_id = engine_id
        self.data_file = data_file
        self.rul_file = rul_file
        self.window_size = 50

        # 状态变量
        self.current_position = 0
        self.window_start_cycle = 0
        self.window_end_cycle = 0
        self.total_cycles = 0
        self.is_running = False
        self.simulation_start_time = None
        self.total_predictions = 0

        # 文件路径
        self.results_file = "simulation_results.csv"
        self.state_file = "simulation_state.json"

        # 预测器
        try:
            if model_path is None:
                # 使用最新的模型文件
                model_files = list(Path("saved_model").glob("xgboost_rul_model_*.pkl"))
                if model_files:
                    model_path = str(max(model_files, key=os.path.getctime))
                else:
                    raise FileNotFoundError("未找到训练好的模型文件")

            self.predictor = RULPredictor(model_path)
            logger.info(f"成功加载模型: {model_path}")

        except Exception as e:
            logger.error(f"加载预测模型失败: {e}")
            raise

        # 数据缓存
        self.engine_data = None
        self.all_data = None

        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"初始化发动机模拟器 - 发动机ID: {engine_id}")

    def load_data(self):
        """加载数据集"""
        logger.info("开始加载数据集...")

        try:
            # 加载测试数据（与其他脚本保持一致的列名）
            columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
                     [f'sensor{i}' for i in range(1, 22)]

            self.all_data = pd.read_csv(self.data_file, sep='\s+', header=None)
            self.all_data.columns = columns
            logger.info(f"原始测试数据: {self.all_data.shape}")

            # 数据预处理（与其他脚本完全一致）
            logger.info("执行数据预处理...")

            # 删除空列
            self.all_data.drop(columns=[26, 27], inplace=True, errors='ignore')

            # 删除常数列
            columns_to_drop = ['sensor1', 'sensor6', 'sensor10', 'sensor16', 'sensor18', 'sensor19']
            self.all_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

            # 传感器名称映射
            sensor_mapping = {
                'sensor2': 'T24', 'sensor3': 'T30', 'sensor4': 'T50', 'sensor7': 'P30',
                'sensor8': 'Nf', 'sensor9': 'Nc', 'sensor11': 'Ps30', 'sensor12': 'phi',
                'sensor13': 'NRf', 'sensor14': 'BPR', 'sensor15': 'htBleed', 'sensor17': 'W31',
                'sensor20': 'W32'
            }

            for old_name, new_name in sensor_mapping.items():
                if old_name in self.all_data.columns:
                    self.all_data.rename(columns={old_name: new_name}, inplace=True)

            logger.info(f"预处理后数据: {self.all_data.shape}")

            # 加载RUL数据
            rul_df = pd.read_csv(self.rul_file, sep='\s+', header=None)
            rul_df.columns = ['RUL']
            rul_df.index = range(1, len(rul_df) + 1)  # 发动机编号从1开始
            logger.info(f"加载RUL数据: {rul_df.shape}")

            # 为测试数据添加RUL（与其他脚本保持一致）
            self.all_data_with_rul = []

            for unit in self.all_data['unit_number'].unique():
                unit_test_data = self.all_data[self.all_data['unit_number'] == unit].copy()
                unit_rul = rul_df.loc[unit, 'RUL']  # RUL DataFrame索引从1开始

                unit_test_data = unit_test_data.sort_values('time_in_cycles')
                max_cycles = unit_test_data['time_in_cycles'].max()
                unit_test_data['RUL'] = unit_rul + max_cycles - unit_test_data['time_in_cycles']

                self.all_data_with_rul.append(unit_test_data)

            self.all_data_with_rul = pd.concat(self.all_data_with_rul, ignore_index=True)
            logger.info(f"处理后的数据: {self.all_data_with_rul.shape}")

            # 获取指定发动机的数据
            self.engine_data = self.all_data_with_rul[
                self.all_data_with_rul['unit_number'] == self.engine_id
            ].copy().sort_values('time_in_cycles')

            self.total_cycles = len(self.engine_data)
            logger.info(f"发动机 {self.engine_id} 数据: {self.total_cycles} 个周期")

            if self.total_cycles < self.window_size:
                raise ValueError(f"发动机 {self.engine_id} 数据点不足 ({self.total_cycles} < {self.window_size})")

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def get_next_window(self):
        """获取下一个时间窗口数据"""
        if self.engine_data is None:
            raise ValueError("数据尚未加载")

        # 检查是否有足够的数据
        if self.current_position + self.window_size > self.total_cycles:
            # 循环到开头
            self.current_position = 0
            logger.info(f"数据遍历完成，重新开始循环 (发动机 {self.engine_id})")

        # 获取窗口数据
        end_position = self.current_position + self.window_size
        window_data = self.engine_data.iloc[self.current_position:end_position].copy()

        # 更新窗口信息
        self.window_start_cycle = window_data['time_in_cycles'].iloc[0]
        self.window_end_cycle = window_data['time_in_cycles'].iloc[-1]

        # 滑动窗口：每次前进1个时间单位（而不是整个窗口长度）
        self.current_position += 1

        logger.debug(f"获取窗口 {self.current_position-1}-{end_position-1}: "
                    f"周期 {self.window_start_cycle}-{self.window_end_cycle}")

        return window_data

    def predict_rul(self, window_data):
        """对窗口数据进行RUL预测"""
        try:
            start_time = time.time()

            # 使用预测器进行预测
            predictions, metadata = self.predictor.predict_rul(window_data)

            processing_time = time.time() - start_time

            if len(predictions) > 0:
                # 使用最后一个预测值（最新周期的预测）
                predicted_rul = predictions[-1]
                logger.info(f"预测完成 - RUL: {predicted_rul:.2f}, 处理时间: {processing_time:.3f}秒")
                return predicted_rul, processing_time
            else:
                logger.warning("预测结果为空")
                return None, processing_time

        except Exception as e:
            logger.error(f"RUL预测失败: {e}")
            return None, 0

    def save_prediction_result(self, predicted_rul, processing_time):
        """保存预测结果到CSV文件"""
        try:
            # 准备结果数据
            result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'engine_id': self.engine_id,
                'window_start_cycle': self.window_start_cycle,
                'window_end_cycle': self.window_end_cycle,
                'predicted_rul': predicted_rul if predicted_rul is not None else 'NULL',
                'model_version': Path(self.predictor.model_path).name if hasattr(self.predictor, 'model_path') else 'unknown',
                'processing_time': processing_time
            }

            # 检查文件是否存在，如果不存在则创建并写入表头
            file_exists = os.path.exists(self.results_file)

            with open(self.results_file, 'a', encoding='utf-8') as f:
                if not file_exists:
                    # 写入表头
                    f.write('timestamp,engine_id,window_start_cycle,window_end_cycle,'
                           'predicted_rul,model_version,processing_time\n')

                # 写入数据
                f.write(f"{result['timestamp']},{result['engine_id']},"
                       f"{result['window_start_cycle']},{result['window_end_cycle']},"
                       f"{result['predicted_rul']},{result['model_version']},"
                       f"{result['processing_time']:.3f}\n")

            logger.debug(f"结果已保存到 {self.results_file}")

        except Exception as e:
            logger.error(f"保存预测结果失败: {e}")

    def save_state(self):
        """保存当前状态"""
        try:
            # 计算下次窗口的预期周期范围
            next_start_cycle = self.window_start_cycle
            next_end_cycle = self.window_end_cycle

            # 如果数据已加载，可以计算下一个窗口的周期
            if self.engine_data is not None and self.current_position < self.total_cycles:
                try:
                    # 预测下一个窗口的位置
                    next_position = self.current_position
                    if next_position + self.window_size <= self.total_cycles:
                        next_window_data = self.engine_data.iloc[next_position:next_position + self.window_size]
                        next_start_cycle = next_window_data['time_in_cycles'].iloc[0]
                        next_end_cycle = next_window_data['time_in_cycles'].iloc[-1]
                except:
                    pass  # 如果计算失败，使用当前值

            state = {
                'engine_id': int(self.engine_id),
                'current_position': int(self.current_position),
                'window_start_cycle': int(next_start_cycle),
                'window_end_cycle': int(next_end_cycle),
                'total_cycles': int(self.total_cycles),
                'last_prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'simulation_start_time': self.simulation_start_time.strftime('%Y-%m-%d %H:%M:%S') if self.simulation_start_time else None,
                'total_predictions': int(self.total_predictions),
                'is_running': bool(self.is_running)
            }

            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)

            logger.debug(f"状态已保存到 {self.state_file}")

        except Exception as e:
            logger.error(f"保存状态失败: {e}")

    def load_state(self):
        """加载之前的状态"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                # 检查是否是同一个发动机
                if state.get('engine_id') == self.engine_id:
                    self.current_position = state.get('current_position', 0)
                    self.total_predictions = state.get('total_predictions', 0)

                    start_time_str = state.get('simulation_start_time')
                    if start_time_str:
                        self.simulation_start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')

                    logger.info(f"恢复状态 - 位置: {self.current_position}, "
                               f"已预测: {self.total_predictions} 次")
                    return True
                else:
                    logger.info(f"状态文件中的发动机ID ({state.get('engine_id')}) "
                               f"与当前发动机ID ({self.engine_id}) 不匹配，使用新状态")
                    return False
            else:
                logger.info("状态文件不存在，使用新状态")
                return False

        except Exception as e:
            logger.error(f"加载状态失败: {e}")
            return False

    def wait_until_next_trigger(self, cron_expression="*/10 * * * *"):
        """等待下一个cron触发时间"""
        try:
            cron = croniter(cron_expression, datetime.now())
            next_time = cron.get_next(datetime)
            wait_seconds = (next_time - datetime.now()).total_seconds()

            if wait_seconds > 0:
                logger.info(f"等待下次触发: {next_time.strftime('%Y-%m-%d %H:%M:%S')} "
                           f"(等待 {wait_seconds:.0f} 秒)")
                time.sleep(wait_seconds)
            else:
                logger.warning("下次触发时间已过，立即执行")

        except Exception as e:
            logger.error(f"等待触发时间失败: {e}")
            # 如果计算失败，等待默认时间
            time.sleep(600)  # 10分钟

    def run_simulation_cycle(self):
        """执行一次模拟周期"""
        try:
            logger.info(f"开始模拟周期 #{self.total_predictions + 1} "
                       f"(发动机 {self.engine_id})")

            # 获取下一个时间窗口
            window_data = self.get_next_window()

            # 进行RUL预测
            predicted_rul, processing_time = self.predict_rul(window_data)

            # 保存预测结果
            self.save_prediction_result(predicted_rul, processing_time)

            # 更新计数器
            self.total_predictions += 1

            # 保存状态
            self.save_state()

            logger.info(f"模拟周期完成 - 窗口: {self.window_start_cycle}-{self.window_end_cycle}, "
                       f"预测RUL: {predicted_rul if predicted_rul is not None else 'NULL'}")

        except Exception as e:
            logger.error(f"模拟周期执行失败: {e}")
            # 即使失败也保存状态
            self.save_state()

    def start_simulation(self, cron_expression="*/10 * * * *"):
        """开始模拟"""
        try:
            logger.info("=" * 60)
            logger.info("启动实时发动机模拟系统")
            logger.info(f"发动机ID: {self.engine_id}")
            logger.info(f"数据文件: {self.data_file}")
            logger.info(f"时间窗口: {self.window_size} 个周期")
            logger.info(f"调度表达式: {cron_expression}")
            logger.info("=" * 60)

            # 加载数据
            self.load_data()

            # 尝试加载之前的状态
            state_loaded = self.load_state()

            # 设置开始时间
            if not state_loaded or self.simulation_start_time is None:
                self.simulation_start_time = datetime.now()
                logger.info(f"开始新的模拟会话: {self.simulation_start_time}")

            self.is_running = True

            # 主循环
            while self.is_running:
                try:
                    # 等待下次触发时间
                    self.wait_until_next_trigger(cron_expression)

                    # 检查是否还在运行
                    if not self.is_running:
                        break

                    # 执行模拟周期
                    self.run_simulation_cycle()

                except KeyboardInterrupt:
                    logger.info("收到中断信号，停止模拟")
                    break
                except Exception as e:
                    logger.error(f"模拟循环出错: {e}")
                    # 等待一段时间后继续
                    time.sleep(60)

        except Exception as e:
            logger.error(f"模拟启动失败: {e}")
            raise
        finally:
            self.is_running = False
            self.save_state()
            logger.info("模拟系统已停止")

    def stop_simulation(self):
        """停止模拟"""
        logger.info("正在停止模拟...")
        self.is_running = False

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"收到信号 {signum}，准备停止模拟...")
        self.stop_simulation()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实时发动机RUL预测模拟系统')
    parser.add_argument('--engine-id', type=int, default=1, help='发动机ID (默认: 1)')
    parser.add_argument('--data-file', default='CMaps/test_FD001.txt', help='测试数据文件')
    parser.add_argument('--rul-file', default='CMaps/RUL_FD001.txt', help='RUL数据文件')
    parser.add_argument('--model-path', help='模型文件路径')
    parser.add_argument('--cron', default='*/10 * * * *', help='Cron表达式 (默认: 每10分钟)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--speedup', type=int, help='时间加速倍数 (仅用于调试)')

    args = parser.parse_args()

    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("启用调试模式")

    try:
        # 创建模拟器
        simulator = RealtimeEngineSimulator(
            engine_id=args.engine_id,
            data_file=args.data_file,
            rul_file=args.rul_file,
            model_path=args.model_path
        )

        # 如果启用加速模式，修改cron表达式
        if args.speedup and args.speedup > 1:
            # 计算加速后的cron表达式
            # 例如：原本每10分钟，加速10倍变成每1分钟
            original_cron = args.cron.split()
            if len(original_cron) >= 1:
                try:
                    original_minute = int(original_cron[0].replace('*/', ''))
                    new_minute = max(1, original_minute // args.speedup)
                    new_cron = f"*/{new_minute} " + " ".join(original_cron[1:])
                    args.cron = new_cron
                    logger.info(f"加速模式 ({args.speedup}x): 调度表达式修改为 {args.cron}")
                except:
                    logger.warning("无法解析cron表达式，使用原始设置")

        # 启动模拟
        simulator.start_simulation(args.cron)

    except KeyboardInterrupt:
        logger.info("用户中断，程序退出")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
# 实时发动机RUL预测模拟系统

## 概述

这是一个实时发动机RUL预测模拟系统，能够模拟真实发动机运行环境，每10分钟滑动时间窗口向前移动一次，对CMaps数据集进行循环遍历和实时RUL预测。

## 🚀 快速开始

### 基本使用

```bash
# 启动默认发动机（ID=1）的实时模拟
python realtime_engine_simulator.py

# 指定发动机ID
python realtime_engine_simulator.py --engine-id 11

# 启用调试模式
python realtime_engine_simulator.py --engine-id 11 --debug
```

### 开发测试模式

```bash
# 加速模式（60倍，用于快速测试）
python realtime_engine_simulator.py --engine-id 11 --debug --speedup 60

# 自定义cron表达式（每5分钟）
python realtime_engine_simulator.py --cron "*/5 * * * *"
```

## 📋 功能特性

### ✅ 已实现功能

- **实时模拟**: 基于cron表达式的定时触发机制
- **循环数据**: 自动循环遍历CMaps数据集
- **状态持久化**: 系统重启后可恢复上次状态
- **结果存储**: CSV格式记录每次预测结果
- **异常处理**: 完善的错误处理和日志记录
- **命令行接口**: 灵活的参数配置
- **多发动机支持**: 可切换不同发动机ID进行模拟

### 📊 输出文件

#### `simulation_results.csv` - 预测结果
```csv
timestamp,engine_id,window_start_cycle,window_end_cycle,predicted_rul,model_version,processing_time
2025-11-04 21:04:42,11,1,50,106.06,xgboost_rul_model_20251104_195207.pkl,0.003
```

#### `simulation_state.json` - 状态信息
```json
{
  "engine_id": 11,
  "current_position": 50,
  "window_start_cycle": 1,
  "window_end_cycle": 50,
  "total_cycles": 83,
  "total_predictions": 1,
  "is_running": false
}
```

#### `simulation.log` - 运行日志
详细的运行日志，包含预测时间、RUL值、处理时间等信息。

## ⚙️ 配置参数

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--engine-id` | 1 | 发动机ID |
| `--data-file` | `CMaps/test_FD001.txt` | 测试数据文件 |
| `--rul-file` | `CMaps/RUL_FD001.txt` | RUL数据文件 |
| `--model-path` | 自动选择最新模型 | 模型文件路径 |
| `--cron` | `*/10 * * * *` | Cron表达式（每10分钟） |
| `--debug` | - | 启用调试模式 |
| `--speedup` | - | 时间加速倍数（仅调试用） |

### Cron表达式示例

```bash
# 每10分钟（默认）
"*/10 * * * *"

# 每5分钟
"*/5 * * * *"

# 每小时
"0 * * * *"

# 每天上午9点
"0 9 * * *"
```

## 🔧 系统架构

### 核心组件

1. **RealtimeEngineSimulator** - 主模拟器类
2. **数据管理器** - 加载和循环遍历CMaps数据
3. **预测引擎** - 集成XGBoost模型进行RUL预测
4. **调度器** - 基于croniter的定时调度
5. **状态管理器** - 状态持久化和恢复
6. **存储器** - CSV格式结果存储

### 工作流程

```
1. 初始化模拟器
2. 加载CMaps数据集
3. 等待cron触发时间
4. 获取下一个时间窗口数据
5. 调用XGBoost模型进行RUL预测
6. 保存预测结果到CSV
7. 更新并保存状态
8. 循环回到步骤3
```

## 📈 使用场景

### 长期监控测试
```bash
# 运行数天或数周的连续模拟
python realtime_engine_simulator.py --engine-id 11
```

### 快速功能验证
```bash
# 5分钟内完成完整测试
python realtime_engine_simulator.py --engine-id 11 --debug --speedup 120
```

### 批量发动机测试
```bash
# 测试多个发动机
for engine_id in 1 5 11 20; do
    python realtime_engine_simulator.py --engine-id $engine_id --speedup 30 &
done
```

## 🛠️ 依赖要求

```bash
pip install croniter pandas numpy joblib xgboost scikit-learn
```

### 项目依赖
- 现有的XGBoost模型文件 (`saved_model/`)
- `predict_rul.py` 预测模块
- CMaps数据集 (`CMaps/`)

## 📊 性能特性

- **内存效率**: 数据预加载，避免重复IO
- **时间精度**: 基于croniter的精确时间控制
- **容错性**: 完善的异常处理，支持自动恢复
- **可扩展**: 模块化设计，易于功能扩展

## 🚨 注意事项

1. **数据要求**: 确保选择的发动机至少有50个数据点
2. **模型文件**: 确保 `saved_model/` 目录中有训练好的模型
3. **磁盘空间**: 长期运行会产生大量CSV数据，注意磁盘空间
4. **系统时间**: 确保系统时间准确，影响cron调度

## 🔍 监控和调试

### 查看实时状态
```bash
# 查看日志
tail -f simulation.log

# 查看最新预测结果
tail -f simulation_results.csv

# 查看当前状态
cat simulation_state.json
```

### 调试模式
```bash
# 启用详细日志
python realtime_engine_simulator.py --debug

# 快速测试（1分钟 = 1秒）
python realtime_engine_simulator.py --speedup 60 --debug
```

## 📝 示例输出

### 控制台输出
```
2025-11-04 21:04:42,780 - INFO - 成功加载模型: saved_model\xgboost_rul_model_20251104_195207.pkl
2025-11-04 21:04:42,781 - INFO - 初始化发动机模拟器 - 发动机ID: 11
2025-11-04 21:04:42,945 - INFO - 发动机 11 数据: 83 个周期
2025-11-04 21:04:42,948 - INFO - 预测完成 - RUL: 106.06, 处理时间: 0.003秒
```

### CSV结果
```csv
timestamp,engine_id,window_start_cycle,window_end_cycle,predicted_rul,model_version,processing_time
2025-11-04 21:04:42,11,1,50,106.06,xgboost_rul_model_20251104_195207.pkl,0.003
2025-11-04 21:14:42,11,2,51,105.89,xgboost_rul_model_20251104_195207.pkl,0.002
```

## 🔮 未来扩展

- [ ] Web界面监控
- [ ] 多发动机并发模拟
- [ ] 预测结果可视化
- [ ] 邮件/短信告警
- [ ] 分布式部署支持
- [ ] 更多数据源支持

## 📞 技术支持

如遇问题，请检查：
1. 模型文件是否存在
2. 数据文件格式是否正确
3. 依赖包是否完整安装
4. 日志文件中的错误信息

更多技术细节请参考：
- `train_xgboost_rul_model.py` - 模型训练脚本
- `predict_rul.py` - 预测模块
- `xgboost-sliding-window-prediction-cn.ipynb` - 研究notebook
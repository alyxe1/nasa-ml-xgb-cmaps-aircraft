# XGBoost RUL预测模型生产部署指南

## 概述

本项目包含基于NASA C-MAPS数据集的XGBoost滑动窗口RUL预测模型的生产级实现。

- **研究验证**: `xgboost-sliding-window-prediction-cn.ipynb` - 研究和验证notebook
- **生产训练**: `train_xgboost_rul_model.py` - 生产级模型训练脚本
- **推理预测**: `predict_rul.py` - 生产级RUL预测脚本

## 模型特点

- **算法**: XGBoost回归器
- **方法**: 滑动窗口时间序列分析
- **窗口大小**: 50个时间周期
- **特征维度**: 850维 (17个传感器特征 × 50个时间窗口)
- **重要传感器**: [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]

## 文件结构

```
saved_model/
├── xgboost_rul_model_20251104_195207.pkl    # 训练好的模型
├── model_metadata_20251104_195207.json      # 模型元数据和性能指标
└── window_params_20251104_195207.pkl        # 滑动窗口参数
```

## 模型性能

最新训练模型 (`xgboost_rul_model_20251104_195207.pkl`):

- **训练集**: MAE=2.16, RMSE=2.78, R²=0.998
- **验证集**: MAE=17.74, RMSE=26.99, R²=0.776
- **测试集**: MAE=25.85, RMSE=35.72, R²=0.532

## 快速开始

### 1. 模型训练

```bash
# 使用默认参数训练
python train_xgboost_rul_model.py

# 自定义参数训练
python train_xgboost_rul_model.py \
    --window-size 50 \
    --n-estimators 300 \
    --learning-rate 0.05 \
    --max-depth 8 \
    --save-dir my_models
```

### 2. 模型推理

```bash
# 预测特定发动机RUL
python predict_rul.py --engine-id 11

# 使用特定模型文件
python predict_rul.py --model-path saved_model/xgboost_rul_model_20251104_195207.pkl --engine-id 1

# 批量预测
python predict_rul.py --data-file your_data.csv
```

### 3. 编程接口

```python
import joblib
import pandas as pd
from predict_rul import RULPredictor

# 加载模型和预测器
predictor = RULPredictor('saved_model/xgboost_rul_model_20251104_195207.pkl')

# 预测单个发动机
result = predictor.predict_engine_rul(your_data, engine_id=11)
print(f"预测RUL: {result['predicted_rul']:.1f}")

# 批量预测
predictions, metadata = predictor.predict_rul(your_data)
```

## 数据格式要求

### 输入数据格式

输入数据应为包含以下列的DataFrame:

**必需列**:
- `unit_number`: 发动机ID
- `time_in_cycles`: 运行周期数
- `operational_setting_1`, `operational_setting_2`, `operational_setting_3`: 操作设置

**传感器列** (必需):
- `sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_8`, `sensor_9`
- `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`
- `sensor_17`, `sensor_20`, `sensor_21`

### 示例数据

```python
import pandas as pd

# 示例数据格式
data = pd.DataFrame({
    'unit_number': [1, 1, 1, 2, 2],
    'time_in_cycles': [1, 2, 3, 1, 2],
    'operational_setting_1': [20.0, 20.1, 20.0, 25.0, 25.1],
    'operational_setting_2': [0.6, 0.6, 0.7, 0.8, 0.8],
    'operational_setting_3': [100.0, 100.0, 99.8, 85.0, 85.2],
    'sensor_2': [518.67, 518.67, 518.67, 521.68, 521.68],
    'sensor_3': [641.82, 641.82, 641.82, 643.64, 643.64],
    # ... 其他传感器数据
})
```

## 模型参数说明

### 训练参数

- `--window-size`: 滑动窗口大小 (默认: 50)
- `--n-estimators`: XGBoost树的数量 (默认: 300)
- `--learning-rate`: 学习率 (默认: 0.05)
- `--max-depth`: 树的最大深度 (默认: 8)
- `--random-state`: 随机种子 (默认: 42)

### 预测参数

- `--batch-size`: 批处理大小，用于大数据集 (默认: 1000)
- `--engine-id`: 预测特定发动机ID
- `--data-file`: 输入数据文件路径

## 注意事项

1. **数据要求**: 每个发动机至少需要50个数据点才能进行预测
2. **特征顺序**: 必须使用与训练时相同的特征列顺序
3. **数据质量**: 确保传感器数据无异常值和缺失值
4. **窗口大小**: 滑动窗口大小必须与训练时一致 (50)

## 性能优化建议

1. **批处理**: 对于大量数据，使用适当的批处理大小
2. **内存管理**: 大数据集建议分批处理
3. **特征预选**: 确保只使用模型训练时的特征列
4. **数据预处理**: 与训练时保持一致的数据预处理步骤

## 故障排除

### 常见错误

1. **特征维度不匹配**: 检查输入数据的特征列是否正确
2. **数据点不足**: 确保每个发动机至少有50个数据点
3. **模型文件不存在**: 检查模型文件路径是否正确

### 调试模式

```bash
# 查看详细输出
python predict_rul.py --engine-id 11 2>&1 | tee debug.log
```

## 技术支持

如需技术支持或有问题反馈，请参考:
- 原始研究notebook: `xgboost-sliding-window-prediction-cn.ipynb`
- 训练脚本: `train_xgboost_rul_model.py`
- 推理脚本: `predict_rul.py`

## 更新日志

- **2025-11-04**: 初始生产版本发布
  - 基于notebook验证结果创建生产级实现
  - 测试集MAE: 25.85, R²: 0.532
  - 支持命令行和编程接口
  - 完整的模型保存和加载功能
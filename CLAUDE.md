# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NASA C-MAPS Turbofan Engine RUL Prediction System using XGBoost with sliding window time series approach. The project implements both research (Jupyter notebooks) and production-ready Python scripts for aircraft engine remaining useful life prediction.

## Core Architecture

### Data Processing Pipeline
- **CMaps Dataset**: NASA Commercial Modular Aero-Propulsion System Simulation data
- **Key Files**: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- **Sensor Selection**: 15 critical sensors from 21 total (T24, T30, T50, P30, Nf, Nc, Ps30, phi, NRf, BPR, htBleed, W31, W32, plus 2 operational settings)
- **Sliding Window**: 50-cycle windows with 1-cycle step (750-dimensional feature vectors: 15×50)

### Core Classes
- **XGBoostRULTrainer** (`train_xgboost_rul_model.py`): Production training pipeline
- **RULPredictor** (`predict_rul.py`): Inference API with model loading
- **RealtimeEngineSimulator** (`realtime_engine_simulator.py`): Real-time simulation with cron scheduling
- **RealtimeEngineSimulatorWebhook** (`realtime_engine_simulator_webhook.py`): Webhook-enabled simulation

## Development Commands

### Model Training
```bash
# Train XGBoost model (production)
python train_xgboost_rul_model.py

# Custom training parameters
python train_xgboost_rul_model.py --window-size 50 --n-estimators 300 --learning-rate 0.05

# Train with custom data directory
python train_xgboost_rul_model.py --data-dir CMaps --output-dir my_models
```

### Prediction & Inference
```bash
# Single engine prediction
python predict_rul.py --engine-id 3

# Batch prediction (multiple engines)
python predict_rul.py --engine-id 1,2,3,5 --output results.csv

# Predict with custom model
python predict_rul.py --model-path saved_model/xgboost_rul_model_20251105_121121.pkl
```

### Real-time Simulation
```bash
# Basic real-time simulation
python realtime_engine_simulator.py --engine-id 3

# Webhook-enabled simulation
python realtime_engine_simulator_webhook.py --engine-id 3 \
  --webhook-url "http://54.238.253.106:5678/webhook-test/e72eda5c-1e4b-4466-8c0b-b3d385119a78"

# Debug mode with time acceleration (for testing)
python realtime_engine_simulator_webhook.py --engine-id 3 --debug --speedup 120
```

### Jupyter Research
```bash
# Main research notebook (Chinese)
jupyter notebook xgboost-sliding-window-prediction-cn.ipynb

# Original research notebook
jupyter notebook damage-propagation-modeling-for-aircraft-engine.ipynb
```

## Key Dependencies

Core libraries (install via pip):
- `pandas`, `numpy` - Data manipulation
- `xgboost` - Gradient boosting algorithm
- `scikit-learn` - ML metrics and utilities
- `matplotlib`, `seaborn` - Visualization
- `croniter` - Cron expression parsing for scheduling
- `requests` - HTTP client for webhooks
- `joblib` - Model serialization

## Data Processing Consistency

**Critical**: All scripts must use identical data preprocessing to maintain feature dimension consistency:

1. **Column Mapping**: sensor2→T24, sensor3→T30, sensor4→T50, sensor7→P30, etc.
2. **Feature Selection**: 15 sensors + 2 operational settings = 17 features
3. **Dropped Columns**: sensor1, sensor5, sensor6, sensor10, sensor16, sensor18, sensor19, plus constant columns
4. **Final Features**: `['setting_1', 'setting_2', 'T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'BPR', 'htBleed', 'W31', 'W32']`

## Model Architecture

### Sliding Window Feature Engineering
- **Window Size**: 50 cycles
- **Step Size**: 1 cycle (not window size - this is critical for proper sliding)
- **Feature Dimension**: 750 (17 features × 50 cycles)
- **Target**: RUL (Remaining Useful Life) at window end

### Model Performance (Latest)
- **Test MAE**: ~26.08
- **Test RMSE**: ~36.11
- **Test R²**: ~0.52

## File Structure Patterns

### Model Files (`saved_model/`)
- `xgboost_rul_model_TIMESTAMP.pkl` - Trained model
- `model_metadata_TIMESTAMP.json` - Training metadata and performance
- `window_params_TIMESTAMP.pkl` - Window parameters and feature configuration

### Simulation Files
- `simulation_results.csv` / `simulation_results_webhook.csv` - Prediction results
- `simulation_state.json` / `simulation_state_webhook.json` - State persistence
- `simulation.log` / `simulation_webhook.log` - Runtime logs

## Webhook Integration

The webhook simulator sends POST requests with JSON payload containing:
- Prediction results and engine info
- Sensor statistics for the current window
- Simulation metadata and performance metrics

Webhook URL format: `http://host:port/webhook-test/{uuid}`

## Important Implementation Details

### Sliding Window Movement
```python
# Correct: Increment by 1 (sliding window)
self.current_position += 1

# Incorrect: Increment by window_size (jumping window)
self.current_position += self.window_size
```

### Model Loading Strategy
- Automatically loads latest model if not specified
- Validates feature dimensions before prediction
- Graceful handling of dimension mismatches with truncation

### State Persistence
- Simulation state survives restarts
- Different engines maintain separate states
- Position tracking ensures continuous operation across sessions

## Testing & Validation

### Test Scripts
- `test_features.py` - Feature dimension validation
- `test_webhook_format.py` - Webhook payload format demonstration

### Debugging Flags
- `--debug` - Enable verbose logging
- `--speedup N` - Accelerate time by N factor for testing
- `--engine-id N` - Specify which engine to simulate

## Common Workflows

### Training New Model
1. Update `train_xgboost_rul_model.py` parameters if needed
2. Run training script
3. Verify model in `saved_model/` directory
4. Test with `predict_rul.py`
5. Optionally test real-time simulation

### Adding New Engines
1. Ensure engine data exists in CMaps dataset
2. Use appropriate engine-id (1-100 for FD001)
3. Test with `predict_rul.py --engine-id N`
4. Run simulation to validate

### Webhook Integration
1. Use `realtime_engine_simulator_webhook.py`
2. Test with `--debug` flag first
3. Verify webhook endpoint accepts POST requests
4. Monitor `simulation_webhook.log` for transmission status
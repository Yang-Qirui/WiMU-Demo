# 推理服务使用说明

## 概述

这是一个基于Flask的推理服务，用于处理设备数据并进行位置估计。服务包含以下功能：

1. **数据预处理**：处理传感器数据
2. **推理API调用**：调用外部推理服务
3. **粒子滤波器管理**：为每个设备维护独立的粒子滤波器
4. **超时检验**：自动检测并重建超时的粒子滤波器

## 服务架构

```
设备数据 → 推理服务 → 预处理 → 推理API → 粒子滤波器 → 位置估计
```

## 启动服务

### 方法1：使用启动脚本（推荐）

```bash
python start_services.py
```

这将同时启动：
- 模拟推理API (端口: 5001)
- 推理服务 (端口: 5002)

### 方法2：手动启动

1. 启动模拟推理API：
```bash
python mock_inference_api.py
```

2. 启动推理服务：
```bash
python backend/inference_service.py
```

## API接口

### 1. 推理接口

**POST** `/inference`

请求体：
```json
{
  "device_id": "1207472511626099",
  "data": {
    "sensor_data": [
      {
        "accel": [0.1, 0.2, 9.8],
        "gyro": [0.01, 0.02, 0.03],
        "timestamp": 1234567890.123
      }
    ]
  }
}
```

响应：
```json
{
  "device_id": "1207472511626099",
  "inference_result": [5.2, 3.1, 1.5],
  "particle_filter_estimate": [5.1, 3.2, 1.4],
  "timestamp": 1234567890.123,
  "status": "success"
}
```

### 2. 健康检查

**GET** `/health`

响应：
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123,
  "active_particle_filters": 2
}
```

### 3. 粒子滤波器状态

**GET** `/particle_filters`

响应：
```json
{
  "1207472511626099": {
    "is_initialized": true,
    "last_update_time": 1234567890.123,
    "is_timeout": false,
    "particle_count": 1000,
    "current_estimate": [5.1, 3.2, 1.4]
  }
}
```

## 测试

运行测试脚本：

```bash
python test_inference_service.py
```

测试内容包括：
- 健康检查
- 推理请求
- 粒子滤波器状态查询
- 超时场景测试

## 配置说明

### 粒子滤波器参数

在 `ParticleFilter` 类中可以调整：

- `num_particles`: 粒子数量（默认1000）
- `timeout_seconds`: 超时时间（默认10秒）
- 粒子分布范围：`low=[0, 0, 0]`, `high=[10, 10, 3]`

### 推理API配置

在 `InferenceService` 类中可以修改：

- `inference_api_url`: 推理API地址
- `timeout`: 请求超时时间（默认30秒）

## 超时机制

- 如果设备的上一次推理请求超过10秒，粒子滤波器会被重置
- 重置时会创建全新的粒子滤波器，确保位置估计的准确性
- 超时检查在每次获取粒子滤波器时自动进行

## 线程安全

- 使用 `threading.Lock` 确保粒子滤波器的线程安全
- 支持多设备并发请求

## 错误处理

- 推理API调用失败时会返回错误信息
- 数据预处理失败时会抛出异常
- 所有异常都会被捕获并返回错误响应

## 日志

服务会输出详细的日志信息，包括：
- 粒子滤波器创建和重置
- 推理请求处理过程
- 错误信息

## 扩展

### 自定义预处理

修改 `preprocess_data` 方法以适应不同的数据格式。

### 自定义粒子滤波器

继承 `ParticleFilter` 类并重写相关方法。

### 自定义推理API

修改 `call_inference_api` 方法以适配不同的推理服务。 
import numpy as np
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """模拟推理API"""
    try:
        data = request.get_json()
        
        # 模拟处理时间
        time.sleep(0.1)
        
        # 模拟推理结果：返回一个随机位置
        position = np.random.uniform(0, 10, 3).tolist()  # [x, y, z]
        
        return jsonify({
            "position": position,
            "confidence": np.random.uniform(0.7, 0.95),
            "timestamp": time.time()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "model": "mock_localization_model",
        "timestamp": time.time()
    })

if __name__ == '__main__':
    print("启动模拟推理API服务...")
    print("地址: http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True) 
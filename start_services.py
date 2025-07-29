import subprocess
import time
import requests
import sys

def start_service(script_name, port, service_name):
    """启动服务"""
    print(f"启动 {service_name}...")
    try:
        process = subprocess.Popen([
            sys.executable, script_name
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待服务启动
        time.sleep(3)
        
        # 检查服务是否启动成功
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ {service_name} 启动成功 (端口: {port})")
                return process
            else:
                print(f"✗ {service_name} 启动失败")
                return None
        except:
            print(f"✗ {service_name} 启动失败")
            return None
            
    except Exception as e:
        print(f"✗ 启动 {service_name} 时出错: {str(e)}")
        return None

def main():
    print("=== 启动推理服务 ===")
    
    # 启动模拟推理API
    mock_api_process = start_service("mock_inference_api.py", 5001, "模拟推理API")
    if not mock_api_process:
        print("模拟推理API启动失败，退出")
        return
    
    # 启动推理服务
    inference_process = start_service("backend/inference_service.py", 5002, "推理服务")
    if not inference_process:
        print("推理服务启动失败，退出")
        mock_api_process.terminate()
        return
    
    print("\n=== 所有服务启动成功 ===")
    print("模拟推理API: http://localhost:5001")
    print("推理服务: http://localhost:5002")
    print("\n按 Ctrl+C 停止所有服务")
    
    try:
        # 保持运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        mock_api_process.terminate()
        inference_process.terminate()
        print("所有服务已停止")

if __name__ == "__main__":
    main() 
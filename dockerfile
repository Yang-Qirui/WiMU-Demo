
# RUN npm install && npm run build

FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime AS backend
WORKDIR /app

# 后端构建阶段
# FROM python:3.12.2 AS python-base

# WORKDIR /app
COPY backend/ ./backend/
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY frontend/dist ./frontend/dist

# 启动
CMD ["python", "backend/backend.py"]

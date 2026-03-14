# Sử dụng Python 3.10 làm base image
FROM python:3.10-slim

# Ngăn Python ghi file .pyc và ép output in thẳng ra terminal (dễ debug)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và MediaPipe chạy trên Linux
# ĐÃ SỬA: Thay libgl1-mesa-glx thành libgl1
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Tạo và đặt thư mục làm việc trong container
WORKDIR /app

# Copy file requirements và cài đặt thư viện Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn dự án vào container
COPY . /app/

# Mở cổng 8000
EXPOSE 8000

# Lệnh khởi động server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
# ---------------
#  Base Image
# ---------------
FROM python:3.10-slim AS builder

WORKDIR /app

# ---------------
#  Install System Dependencies
# ---------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------------
#  Install Python Dependencies
# ---------------
COPY requirements.txt .

#no-cache-dir to reduce image size by not caching the packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------
#  Application Code
# ---------------
COPY . .

# -----------------
#  Expose Ports
# -----------------
# Pending: Web server (FastAPI), to expose the port
# EXPOSE 8000

# -----------------
#  Command
# -----------------
# Command to run the application by default
# docker-compose will override this command if needed
# CMD ["python", "src/main.py"]
# --- Simple English summary ---
# A Dockerfile is a recipe that packages this whole app (code + Python +
# libraries) into one "container image" that runs the same way on any
# machine — your laptop, a teammate's laptop, or a cloud server.

# Base image — Python 3.11 slim version for smaller image size
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed for compiling certain Python database packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Create necessary directories inside container
RUN mkdir -p mlruns ml monitoring logs

# Expose port 8000 for FastAPI
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Start FastAPI app using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
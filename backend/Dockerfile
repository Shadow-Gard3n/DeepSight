FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (HF expects 7860)
EXPOSE 7860

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
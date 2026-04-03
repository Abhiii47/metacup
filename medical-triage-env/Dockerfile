FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the API port (Hugging Face Spaces default is 7860)
EXPOSE 7860

# Set default environment variables
ENV ENV_DIFFICULTY=easy
ENV PYTHONUNBUFFERED=1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

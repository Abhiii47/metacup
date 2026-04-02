FROM python:3.10-slim

WORKDIR /app

# Copy the requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the API and WebSocket port (Hugging Face spaces default is 7860)
EXPOSE 7860

# Command to run the FastAPI server using uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

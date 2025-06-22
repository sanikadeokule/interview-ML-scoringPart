FROM python:3.11-slim

# Install ffmpeg and other system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose port (match your Flask app)
EXPOSE 10000

# Start the Flask app
CMD ["python", "newfile.py"]
# Use Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system packages (like ffmpeg) and clean up to reduce image size
RUN apt-get update && \
    apt-get install --no-install-recommends -y ffmpeg 

# Copy the requirements file
COPY requirements_no_pywin32.txt ./

# Upgrade pip and install dependencies (this will use the host-mounted cache)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_no_pywin32.txt

# Copy the application code
COPY . .

# Expose the port Gunicorn will run on
EXPOSE 8000

# Define environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Command to run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]

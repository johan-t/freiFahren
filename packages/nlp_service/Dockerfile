# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container to /app
WORKDIR /app

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Add the current directory contents into the container at /app
ADD . /app

# Install the required packages using the setup.py in editable mode
RUN pip install --no-cache-dir -e nlp_service/

# Copy the .env file into the container
COPY .env /app/.env

# Run the command to start the main application
CMD ["python3", "-m", "nlp_service.main"]

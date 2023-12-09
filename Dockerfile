FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /src

# Copy the requirements.txt file and install the required packages
COPY requirements.txt requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file and install the required packages
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the source files and the entrypoint script
COPY src/ /src/

# Make the entrypoint script executable
RUN chmod +x /src/entrypoint.sh

# Expose the Jupyter Notebook port
EXPOSE 8888

# Run the entrypoint script
CMD ["/src/entrypoint.sh"]

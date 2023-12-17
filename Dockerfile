FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /src

# Copy the requirements.txt file and install the required packages
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

#store hf models in this persisted space
ENV TRANSFORMERS_CACHE=/src/data/model-cache/

# Expose the Jupyter Notebook port
EXPOSE 8888
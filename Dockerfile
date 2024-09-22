FROM nvidia/cuda:12.2.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3.10-venv python3.10-dev git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN ls -la /usr/bin/python3 && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy the Flask application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Set the entrypoint to run the Flask application
ENTRYPOINT ["gunicorn", "-b", ":8080", "app:app"]
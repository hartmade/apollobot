FROM python:3.12-slim

WORKDIR /app

# System dependencies for numpy/scipy/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev liblapack-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy and install
COPY . /app
RUN pip install --no-cache-dir -e ".[notifications]"

# Create directories
RUN mkdir -p /root/.apollobot /root/apollobot-research

# Volumes for persistent output
VOLUME ["/root/.apollobot", "/root/apollobot-research"]

ENTRYPOINT ["apollo"]

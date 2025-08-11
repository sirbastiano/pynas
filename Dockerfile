# Use an official Python runtime as a parent image
FROM python:3.9-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*



# Copy dependency definition files
COPY pyproject.toml pdm.lock /app/
COPY pynas /app/pynas
COPY notebooks /app/notebooks
COPY datasets /app/datasets
COPY scripts /app/scripts
COPY pyscripts /app/pyscripts


# Install project dependencies using PDM, excluding development dependencies
# Install PDM for package management
RUN pip install --no-cache-dir pdm \
    && pdm install --prod --no-lock

# The command to run the application will be specified in docker-compose.yml
# or can be added here. For example:
# CMD ["pdm", "run", "python3", "nas_seg_burned.py"]

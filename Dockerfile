# Start from a CUDA-enabled base image
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip

# Install PyTorch Hugging Face Transformers and other necessary packages
RUN pip3 install pdm

# Create an output folder
RUN mkdir /output

COPY pdm.lock pyproject.toml /app/

RUN pdm install

COPY download_model.py /app/

RUN pdm run python download_model.py

# Copy your Python script into the container
COPY src/compute_horde_prompt_solver /app/compute_horde_prompt_solver

# Set the entrypoint to run your script
ENTRYPOINT ["pdm", "run", "python", "/app/compute_horde_prompt_solver/run.py", "--model=/app/saved_model/", "--output-dir=/output"]

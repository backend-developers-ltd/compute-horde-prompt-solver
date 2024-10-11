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
# RUN pdm run pip install vllm

RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    python3 use_existing_torch.py && \
    cd ..

RUN pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
RUN pdm add setuptools-scm
RUN cd vllm && \
    sed -i 's/^    os.symlink(current_vllm_path, pre_built_vllm_path)/#    os.symlink(current_vllm_path, pre_built_vllm_path)/' python_only_dev.py && \
    sed -i '/os.rename(pre_built_vllm_path, tmp_path)/s/^/#/' python_only_dev.py && \
    python3 python_only_dev.py && \
    cd ..
RUN pdm add pip
RUN pdm run python -mpip install -r vllm/requirements-build.txt
RUN . .venv/bin/activate && cd vllm && pip install -r requirements-cuda.txt && cd ..
RUN pdm run python -c 'import psutil'
RUN pdm run python -mpip install --no-deps https://files.pythonhosted.org/packages/4a/4c/ee65ba33467a4c0de350ce29fbae39b9d0e7fcd887cc756fa993654d1228/vllm-0.6.3.post1-cp38-abi3-manylinux1_x86_64.whl
RUN pdm run python -mpip install compressed-tensors==0.6.0
RUN pdm run python -mpip install xformers==0.0.27.post2 --no-deps
RUN pdm run python -c 'import vllm'

COPY download_model.py /app/

RUN pdm run python download_model.py

# Copy your Python script into the container
COPY src/compute_horde_prompt_solver /app/compute_horde_prompt_solver

# Set the entrypoint to run your script
ENTRYPOINT ["pdm", "run", "python", "/app/compute_horde_prompt_solver/run.py", "--model=/app/saved_model/", "--output-dir=/output"]

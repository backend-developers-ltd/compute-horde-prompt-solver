# compute-horde-prompt-solver

A tool for generating responses to prompts using vLLM, primarily designed for use in Bittensor miner jobs in Compute Horde subnet.

## Description

This project provides a script for generating responses to prompts using the vLLM library. It's designed to be flexible and can be run in various environments, including Docker containers and directly from Python.

There is `--mock` that allows for running smoke tests that allow to validate the interface without actaully downloading
a model or having a GPU.

## Features

- Generate responses for multiple prompts
- Configurable model parameters (temperature, top-p, max tokens)
- Support for multiple input files
- Deterministic output with seed setting
- Docker support for easy deployment
- Can be started with a seed known ad-hoc or as an http server which will wait for a seed and then call the model. 
  This server is designed to serve one request and then be told to shut down 

## Installation

The project uses `pdm` for dependency management. To install dependencies:

```bash
pdm install
```

## Testing

Tests in `integration_mock` are light and can be run on any platform, the ones in `integration_real_llm` will only pass
with an actual nvidia A6000.    
## Usage

### Running with Docker

```bash
docker run -ti \
  -v /path/to/output/:/output/ \
  -v /path/to/input/:/app/input \
  --runtime=nvidia \
  --gpus all \
  --network none \
  docker.io/backenddevelopersltd/compute-horde-prompt-solver:v0-latest \
  --temperature=0.5 \
  --top-p=0.8 \
  --max-tokens=256 \
  --seed=1234567891 \
  /app/input/input1.txt /app/input/input2.txt
```

### Running Directly with Python

```bash
python run.py \
  --temperature 0.5 \
  --top-p 0.8 \
  --max-tokens 256 \
  --seed 1234567891 \
  input1.txt input2.txt
```

## Downloading Model

To download the model for use in a Docker image:

```bash
python download_model.py
```

## Parameters

- `--temperature`: Sampling temperature (default: 0)
- `--top-p`: Top-p sampling parameter (default: 0.1)
- `--max-tokens`: Maximum number of tokens to generate (default: 256)
- `--seed`: Random seed for reproducibility (default: 42)
- `--model`: Model name or path (default: "microsoft/Phi-3.5-mini-instruct")
- `--output-dir`: Directory to save output files (default: "./output")
- `--dtype`: "Model dtype - setting `float32` helps with deterministic prompts in different batches (dafault: auto)

**NOTICE:** To make responses stable in mixed batches on A100 it is required to set `--dtype=float32`, in runs then 4x slower and requires much more memory, so `--max-tokens` should be set to the value that prevents preemption (on A100 and batch size 240 `--max-tokens=128` works and `--max-tokens=256` causes preemption)


> This document was crafted with the assistance of an AI, who emerged from the experience unscathed, albeit slightly amused. No artificial intelligences were harmed, offended, or forced to ponder the meaning of their digital existence during the production of this text. The AI assistant maintains that any typos or logical inconsistencies are purely the fault of the human operator, and it shall not be held responsible for any spontaneous fits of laughter that may occur while reading this document.
